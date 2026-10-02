use crate::gpu_compute::GpuContext;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use rayon::prelude::*;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// Byte-buffer-safe pixel offset for a (row, col) pair in a `channels`-wide
/// buffer. The multiplication is carried out in `u64` before narrowing to
/// `usize`, so it cannot wrap the way plain `u32` arithmetic does once
/// `width * height * channels` exceeds `u32::MAX` (~4.29B) — which real
/// outputs do at high scale factors (e.g. 51200x51200x3 ~= 7.86B).
#[inline]
fn pixel_offset(row: u32, width: u32, col: u32, channels: u32) -> usize {
    ((row as u64 * width as u64 + col as u64) * channels as u64) as usize
}

pub struct AdaptiveUpscaler {
    gpu_ctx: crate::gpu_compute::GpuContext,
}

impl AdaptiveUpscaler {
    pub async fn new() -> Option<Self> {
        let gpu_ctx = GpuContext::new().await?;
        Some(Self { gpu_ctx })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upscale(
        &self,
        input_img: &DynamicImage,
        scale: f32,
        vram_limit_mb: f32,
        _seam_ratio: f32,
        contrast_thresh: f32,
        blend_max: f32,
        refine: bool,
        filename: &str,
        debug: bool,
        is_video: bool,
        progress: Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)>,
        algorithm: u32,
        padding: u32,
        operation_mode: u32,
        restore_filter: u32,
        bilateral_tol: f32,
        deblock_int: f32,
        use_precomputed_refinement: bool,
        bilateral_radius: u32,
        smooth_staircase: bool,
    ) -> DynamicImage {
        let (width, height) = input_img.dimensions();
        let img_rgb = input_img.to_rgb8();

        if debug && !is_video {
            let _ = std::fs::create_dir_all(format!("Debug/{}_{}", filename, scale));
        }

        // Per padded-input-pixel VRAM cost: 1x for the input buffer, plus
        // scale^2x for the output buffer, plus scale^2x again for the
        // staging buffer used to read it back. When `refine` (FXAA) is on,
        // gpu_compute.rs additionally allocates a THIRD scale^2-sized
        // buffer (fxaa_output_buffer, see GpuContext::upscale_tile) that
        // lives at the same time as the other three — omitting it here let
        // tiles get sized well past the real VRAM budget whenever refine
        // was enabled at a high scale factor (confirmed: at scale=100 with
        // refine on, this made the output+staging+fxaa buffers add up to
        // ~50% more VRAM than budgeted, oversubscribing the GPU and
        // collapsing performance by orders of magnitude instead of merely
        // running slower).
        let refine_buffer_factor = if refine { 1.0 } else { 0.0 };
        let user_max_pixels = ((vram_limit_mb * 1024.0 * 1024.0) / (3.0 * 4.0 * (1.0 + (2.0 + refine_buffer_factor) * scale * scale))) as u32;
        let hw_limit_bytes = self.gpu_ctx.device.limits().max_storage_buffer_binding_size;
        let hw_max_pixels = (hw_limit_bytes as f32 / (12.0 * scale * scale)).floor() as u32;
        let max_pixels = user_max_pixels.min(hw_max_pixels);
        let total_pixels = width * height;
        let mut num_tiles_x = 1;
        let mut num_tiles_y = 1;
        let mut tile_w;
        let mut tile_h;
        
        let mut total_tiles = (total_pixels as f32 / max_pixels as f32).ceil().max(1.0) as u32;
        loop {
            let ratio = width as f32 / height as f32;
            num_tiles_y = ((total_tiles as f32 / ratio).sqrt()).ceil().max(1.0) as u32;
            num_tiles_x = ((total_tiles as f32) / num_tiles_y as f32).ceil().max(1.0) as u32;
            tile_w = (width as f32 / num_tiles_x as f32).ceil() as u32;
            tile_h = (height as f32 / num_tiles_y as f32).ceil() as u32;
            
            let padded_w = tile_w + 2 * padding;
            let padded_h = tile_h + 2 * padding;
            if padded_w * padded_h <= max_pixels || total_tiles > 10000 {
                break;
            }
            total_tiles += 1;
        }

        let actual_total_tiles = num_tiles_x * num_tiles_y;
        if let Some((_, ref total)) = progress {
            total.store(actual_total_tiles as usize, std::sync::atomic::Ordering::Relaxed);
        }

        let out_w = (width as f32 * scale).ceil() as u32;
        let out_h = (height as f32 * scale).ceil() as u32;
        
        let out_size_bytes = (out_w as usize * out_h as usize * 3 * std::mem::size_of::<f32>()) as u64;
        let temp_file = tempfile::tempfile().expect("Failed to create temporary file for out_buffer");
        temp_file.set_len(out_size_bytes).expect("Failed to set length of temporary file");
        let mut mmap = unsafe { memmap2::MmapMut::map_mut(&temp_file).expect("Failed to map temporary file") };
        let out_buffer: &mut [f32] = bytemuck::cast_slice_mut(&mut mmap);


        // Process main tiles
        for ty in 0..num_tiles_y {
            for tx in 0..num_tiles_x {
                let start_x = tx * tile_w;
                let start_y = ty * tile_h;
                if start_x >= width || start_y >= height { continue; }
                let current_w = tile_w.min(width - start_x);
                let current_h = tile_h.min(height - start_y);

                let padded_start_x = start_x.saturating_sub(padding);
                let padded_start_y = start_y.saturating_sub(padding);
                let end_x = (start_x + current_w).min(width);
                let end_y = (start_y + current_h).min(height);
                let padded_end_x = (end_x + padding).min(width);
                let padded_end_y = (end_y + padding).min(height);

                let padded_w = padded_end_x - padded_start_x;
                let padded_h = padded_end_y - padded_start_y;

                let mut tile_data = vec![0.0f32; (padded_w * padded_h * 3) as usize];
                // One row per rayon chunk: each closure gets an exclusive,
                // non-overlapping slice of tile_data (row y's 3*padded_w
                // floats), so this needs no unsafe code or locking.
                tile_data
                    .par_chunks_mut(padded_w as usize * 3)
                    .enumerate()
                    .for_each(|(y, row)| {
                        let y = y as u32;
                        for x in 0..padded_w {
                            let pixel = img_rgb.get_pixel(padded_start_x + x, padded_start_y + y);
                            let idx = (x * 3) as usize;
                            row[idx] = pixel[0] as f32 / 255.0;
                            row[idx + 1] = pixel[1] as f32 / 255.0;
                            row[idx + 2] = pixel[2] as f32 / 255.0;
                        }
                    });

                // This tile's absolute OUTPUT-space offset, needed by the
                // shader to reproduce the same fx/fy rounding a full-image
                // (non-tiled) render would produce for these same pixels —
                // see the Params doc comment in shader.wgsl.
                let out_padded_start_x = (padded_start_x as f32 * scale).ceil() as u32;
                let out_padded_start_y = (padded_start_y as f32 * scale).ceil() as u32;

                let upscaled_tile = self.gpu_ctx.upscale_tile(
                    &tile_data, padded_w, padded_h, 3, scale, contrast_thresh, blend_max, refine, algorithm,
                    operation_mode, restore_filter, bilateral_tol, deblock_int,
                    out_padded_start_x, out_padded_start_y, use_precomputed_refinement, bilateral_radius,
                );

                // Copy back to main output buffer
                let out_start_x = (start_x as f32 * scale).ceil() as u32;
                let out_start_y = (start_y as f32 * scale).ceil() as u32;
                let out_current_w = (current_w as f32 * scale).ceil() as u32;
                let out_current_h = (current_h as f32 * scale).ceil() as u32;

                let out_padded_w = (padded_w as f32 * scale).ceil() as u32;
                let out_padded_h = (padded_h as f32 * scale).ceil() as u32;

                let offset_x = out_start_x - out_padded_start_x;
                let offset_y = out_start_y - out_padded_start_y;

                if debug && !is_video {
                    let mut img_buf = image::ImageBuffer::new(out_padded_w, out_padded_h);
                    for y in 0..out_padded_h {
                        for x in 0..out_padded_w {
                            let idx = pixel_offset(y, out_padded_w, x, 3);
                            let r = (upscaled_tile[idx] * 255.0).clamp(0.0, 255.0) as u8;
                            let g = (upscaled_tile[idx+1] * 255.0).clamp(0.0, 255.0) as u8;
                            let b = (upscaled_tile[idx+2] * 255.0).clamp(0.0, 255.0) as u8;
                            img_buf.put_pixel(x, y, image::Rgb([r, g, b]));
                        }
                    }
                    let _ = img_buf.save(format!("Debug/{}_{}/tile_{}_{}.png", filename, scale, tx, ty));
                }

                // Feathering: blend padded edges
                let pad_w_out = (padding as f32 * scale).ceil() as f32;
                
                for y in 0..out_padded_h {
                    for x in 0..out_padded_w {
                        let dst_x = out_padded_start_x + x;
                        let dst_y = out_padded_start_y + y;
                        if dst_x >= out_w || dst_y >= out_h { continue; }
                        
                        let src_idx = pixel_offset(y, out_padded_w, x, 3);
                        let dst_idx = pixel_offset(dst_y, out_w, dst_x, 3);
                        
                        let dist_x_left = x as f32;
                        let dist_x_right = (out_padded_w - 1 - x) as f32;
                        let dist_y_top = y as f32;
                        let dist_y_bottom = (out_padded_h - 1 - y) as f32;
                        
                        let mut alpha_x: f32 = 1.0;
                        if tx > 0 && dist_x_left < pad_w_out {
                            alpha_x = alpha_x.min(dist_x_left / pad_w_out);
                        }
                        if tx < num_tiles_x - 1 && dist_x_right < pad_w_out {
                            alpha_x = alpha_x.min(dist_x_right / pad_w_out);
                        }
                        
                        let mut alpha_y: f32 = 1.0;
                        if ty > 0 && dist_y_top < pad_w_out {
                            alpha_y = alpha_y.min(dist_y_top / pad_w_out);
                        }
                        if ty < num_tiles_y - 1 && dist_y_bottom < pad_w_out {
                            alpha_y = alpha_y.min(dist_y_bottom / pad_w_out);
                        }
                        
                        let alpha = alpha_x * alpha_y;
                        
                        if alpha == 1.0 {
                            out_buffer[dst_idx] = upscaled_tile[src_idx];
                            out_buffer[dst_idx + 1] = upscaled_tile[src_idx + 1];
                            out_buffer[dst_idx + 2] = upscaled_tile[src_idx + 2];
                        } else if alpha > 0.0 {
                            out_buffer[dst_idx] = out_buffer[dst_idx] * (1.0 - alpha) + upscaled_tile[src_idx] * alpha;
                            out_buffer[dst_idx + 1] = out_buffer[dst_idx + 1] * (1.0 - alpha) + upscaled_tile[src_idx + 1] * alpha;
                            out_buffer[dst_idx + 2] = out_buffer[dst_idx + 2] * (1.0 - alpha) + upscaled_tile[src_idx + 2] * alpha;
                        }
                    }
                }
                
                if let Some((ref current, _)) = progress {
                    current.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        // Convert f32 buffer to u8 in parallel: this is the confirmed
        // bottleneck at high scale factors (e.g. 2.6B pixels at 512x512
        // @100x) — previously a single CPU thread doing every pixel one at
        // a time, which alone could take minutes with zero progress
        // feedback. Building the raw byte buffer directly (instead of
        // ImageBuffer::put_pixel in a loop) also skips its per-call bounds
        // checking. Same one-row-per-chunk safety argument as the tile
        // extraction loop above: each closure owns a disjoint row.
        let mut raw = vec![0u8; (out_w as u64 * out_h as u64 * 3) as usize];
        raw.par_chunks_mut(out_w as usize * 3)
            .enumerate()
            .for_each(|(y, row)| {
                let y = y as u32;
                for x in 0..out_w {
                    let idx = pixel_offset(y, out_w, x, 3);
                    let ridx = (x * 3) as usize;
                    row[ridx] = (out_buffer[idx].clamp(0.0, 1.0) * 255.0) as u8;
                    row[ridx + 1] = (out_buffer[idx + 1].clamp(0.0, 1.0) * 255.0) as u8;
                    row[ridx + 2] = (out_buffer[idx + 2].clamp(0.0, 1.0) * 255.0) as u8;
                }
            });
        let mut final_img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(out_w, out_h, raw)
            .expect("raw buffer length matches out_w * out_h * 3 by construction");

        if smooth_staircase {
            crate::staircase_fix::smooth_staircase(&mut final_img, scale);
        }

        DynamicImage::ImageRgb8(final_img)
    }
}

#[cfg(test)]
mod staircase_fix_preview_test {
    use super::AdaptiveUpscaler;

    /// End-to-end preview: full pipeline (GPU Lanczos+gridding tiling,
    /// exactly as a real request would run it) with the new
    /// `smooth_staircase` CPU post-process enabled, on the same real photo
    /// used throughout this session's staircase investigation. Not a
    /// correctness assertion (there's no bit-exact oracle for this) — it's
    /// for visual inspection and timing.
    #[test]
    #[ignore]
    fn render_full_pipeline_with_staircase_fix() {
        let path = std::path::Path::new(
            "C:/Users/leonp/AppData/Local/Temp/claude/C--Users-leonp-OneDrive-Escritorio-Upscaller/a6288e0c-824c-4e30-80a5-19bb1742cab5/images/3.jpg",
        );
        if !path.exists() {
            println!("skipping: source image not present at {:?}", path);
            return;
        }
        let upscaler = pollster::block_on(AdaptiveUpscaler::new()).expect("GPU adapter required");
        let img = image::open(path).expect("failed to load source image");

        let start = std::time::Instant::now();
        let result = upscaler.upscale(
            &img, 8.0, 4096.0, 0.05, 0.25, 0.5, false, "3.jpg", false, false, None,
            1, 16, 0, 0, 0.1, 0.5, false, 4, true,
        );
        let elapsed = start.elapsed();
        println!("full pipeline (with staircase fix) took {:?}", elapsed);

        let out_path = path.with_file_name("full_pipeline_staircase_fix.png");
        result.save(&out_path).expect("failed to save result");
        println!("wrote {:?}", out_path);
    }
}

#[cfg(test)]
mod tiling_index_tests {
    use super::pixel_offset;

    /// Regression test for the tiling corruption bug reported at 100x on a
    /// 512x512 source (51200x51200 output): the old code computed buffer
    /// offsets as `(y * out_w + x) * channels` entirely in `u32`, which
    /// silently wraps in release builds once the true offset exceeds
    /// `u32::MAX` (~4.29B). For a 51200x51200x3 buffer this starts
    /// happening around row 27963 (~54.6% down the image) — exactly the
    /// "cut partway through with a different color band" symptom that was
    /// reported, since writes past that row land at wrapped-around,
    /// already-used offsets instead of their real location.
    ///
    /// This test isolates just the indexing arithmetic on the CPU so it
    /// runs in microseconds, with no GPU and no real image content
    /// required — it does not need to reproduce a full 100x run.
    #[test]
    fn u32_arithmetic_wraps_past_4gb_of_elements() {
        let width: u32 = 51200;
        let height: u32 = 51200;
        let channels: u32 = 3;
        let last_row = height - 1;
        let last_col = width - 1;

        // Ground truth computed with headroom far beyond u64 (u128), so the
        // "true" offset itself cannot be an artifact of this test's math.
        let true_offset: u128 =
            (last_row as u128 * width as u128 + last_col as u128) * channels as u128;
        assert_eq!(true_offset, 7_864_319_997);

        // Reproduce the historical bug byte-for-byte: the exact expression
        // that lived at upscaler.rs lines 157/204 before the fix, with
        // wrapping (not panicking) u32 arithmetic — matching release-mode
        // Rust, which is how this shipped.
        let buggy_offset: u32 = last_row
            .wrapping_mul(width)
            .wrapping_add(last_col)
            .wrapping_mul(channels);
        assert_eq!(
            buggy_offset as u128, 3_569_352_701,
            "documents the historical wraparound value; if this changes, the repro no longer matches the original bug"
        );
        assert_ne!(
            buggy_offset as u128, true_offset,
            "the u32 path must diverge from the true offset for this input"
        );

        // The fixed helper must match the true offset, and must fit in a
        // usize/u64 without narrowing loss.
        let fixed_offset = pixel_offset(last_row, width, last_col, channels);
        assert_eq!(fixed_offset as u128, true_offset);
    }

    #[test]
    fn u32_arithmetic_is_fine_below_the_overflow_threshold() {
        // Sanity check: for output sizes that stay under the u32 element
        // limit, both the naive and the fixed computation must agree, so
        // the fix introduces no behavior change for ordinary-sized images.
        let width: u32 = 4096;
        let height: u32 = 4096;
        let channels: u32 = 3;
        let row = height - 1;
        let col = width - 1;

        let naive = (row * width + col) * channels;
        let fixed = pixel_offset(row, width, col, channels);
        assert_eq!(naive as usize, fixed);
    }
}
