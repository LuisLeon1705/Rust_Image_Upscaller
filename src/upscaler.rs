use crate::gpu_compute::GpuContext;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

pub struct AdaptiveUpscaler {
    gpu_ctx: crate::gpu_compute::GpuContext,
}

impl AdaptiveUpscaler {
    pub async fn new() -> Option<Self> {
        let gpu_ctx = GpuContext::new().await?;
        Some(Self { gpu_ctx })
    }

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
    ) -> DynamicImage {
        let (width, height) = input_img.dimensions();
        let img_rgb = input_img.to_rgb8();

        if debug && !is_video {
            let _ = std::fs::create_dir_all(format!("Debug/{}_{}", filename, scale));
        }

        let user_max_pixels = ((vram_limit_mb * 1024.0 * 1024.0) / (3.0 * 4.0 * (1.0 + 2.0 * scale * scale))) as u32;
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
                for y in 0..padded_h {
                    for x in 0..padded_w {
                        let pixel = img_rgb.get_pixel(padded_start_x + x, padded_start_y + y);
                        let idx = ((y * padded_w + x) * 3) as usize;
                        tile_data[idx] = pixel[0] as f32 / 255.0;
                        tile_data[idx + 1] = pixel[1] as f32 / 255.0;
                        tile_data[idx + 2] = pixel[2] as f32 / 255.0;
                    }
                }

                let upscaled_tile = self.gpu_ctx.upscale_tile(
                    &tile_data, padded_w, padded_h, 3, scale, contrast_thresh, blend_max, refine, algorithm,
                    operation_mode, restore_filter, bilateral_tol, deblock_int
                );

                // Copy back to main output buffer
                let out_start_x = (start_x as f32 * scale).ceil() as u32;
                let out_start_y = (start_y as f32 * scale).ceil() as u32;
                let out_current_w = (current_w as f32 * scale).ceil() as u32;
                let out_current_h = (current_h as f32 * scale).ceil() as u32;

                let out_padded_start_x = (padded_start_x as f32 * scale).ceil() as u32;
                let out_padded_start_y = (padded_start_y as f32 * scale).ceil() as u32;
                let out_padded_w = (padded_w as f32 * scale).ceil() as u32;
                let out_padded_h = (padded_h as f32 * scale).ceil() as u32;

                let offset_x = out_start_x - out_padded_start_x;
                let offset_y = out_start_y - out_padded_start_y;

                if debug && !is_video {
                    let mut img_buf = image::ImageBuffer::new(out_padded_w, out_padded_h);
                    for y in 0..out_padded_h {
                        for x in 0..out_padded_w {
                            let idx = ((y * out_padded_w + x) * 3) as usize;
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
                        
                        let src_idx = ((y * out_padded_w + x) * 3) as usize;
                        let dst_idx = ((dst_y * out_w + dst_x) * 3) as usize;
                        
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

        // Convert f32 buffer to ImageBuffer
        let mut final_img = ImageBuffer::new(out_w, out_h);
        for y in 0..out_h {
            for x in 0..out_w {
                let idx = ((y * out_w + x) * 3) as usize;
                let r = (out_buffer[idx].clamp(0.0, 1.0) * 255.0) as u8;
                let g = (out_buffer[idx + 1].clamp(0.0, 1.0) * 255.0) as u8;
                let b = (out_buffer[idx + 2].clamp(0.0, 1.0) * 255.0) as u8;
                final_img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        let dyn_img = DynamicImage::ImageRgb8(final_img);

        dyn_img
    }
}
