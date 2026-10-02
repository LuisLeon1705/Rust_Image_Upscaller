use wgpu::util::DeviceExt;
use std::borrow::Cow;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub scale: f32,
    pub contrast_thresh: f32,
    pub blend_max: f32,
    pub algorithm: u32,
    pub operation_mode: u32,
    pub restore_filter: u32,
    pub bilateral_tol: f32,
    pub deblock_int: f32,
    /// This tile's absolute OUTPUT-space (x, y) offset in the final image.
    /// Only consumed by shader.wgsl's main() — see the doc comment on its
    /// Params struct for why this exists (tile-addressing fix, not a
    /// resampling/quality change). Unused by restoration.wgsl and
    /// fxaa.wgsl, which don't declare these trailing fields at all; a
    /// uniform buffer binding only needs to be at least as large as what a
    /// shader's own struct declares, so leaving them zeroed for those two
    /// pipelines is harmless.
    pub origin_x: u32,
    pub origin_y: u32,
    /// EXPERIMENTAL toggle, off (0) by default — see the doc comment on
    /// this field in shader.wgsl's Params struct and refine_precompute.wgsl.
    /// Only consumed by shader.wgsl's main(); restoration.wgsl and
    /// fxaa.wgsl don't declare this trailing field so they never read it.
    pub use_precomputed_refinement: u32,
    /// Spatial radius (in pixels) of the bilateral restoration filter's
    /// window — see the doc comment on `apply_bilateral` in restoration.wgsl.
    /// A bigger radius gives the filter more samples to average smoothly
    /// along a diagonal edge instead of quantizing it into a staircase; a
    /// smaller radius is cheaper and preserves more fine texture. Only
    /// consumed by restoration.wgsl.
    pub bilateral_radius: u32,
    pub _pad5: u32,
}

#[cfg(test)]
mod params_layout_test {
    use super::Params;

    /// WGSL requires a struct used directly as a `var<uniform>` binding's
    /// type to have a size that's a multiple of 16 bytes. Getting this
    /// wrong doesn't reliably fail loudly (wgpu's validation layer is
    /// commonly off in release builds) — it can silently read past the
    /// buffer instead, which is exactly what caused the origin_x/origin_y
    /// tile-addressing fix to read garbage the first time this was wired
    /// up. Pin this down as a compile-time-checked invariant so it can
    /// never silently regress again.
    #[test]
    fn params_size_is_multiple_of_16_bytes() {
        assert_eq!(
            std::mem::size_of::<Params>() % 16,
            0,
            "Params must stay a multiple of 16 bytes for WGSL uniform buffer layout — add/remove a _padN field"
        );
    }
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub compute_pipeline: wgpu::ComputePipeline,
    pub fxaa_pipeline: wgpu::ComputePipeline,
    pub restoration_pipeline: wgpu::ComputePipeline,
    /// EXPERIMENTAL, toggleable — see refine_precompute.wgsl.
    pub refine_precompute_pipeline: wgpu::ComputePipeline,
}

impl GpuContext {
    pub async fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await?;
        let limits = adapter.limits();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            ..Default::default()
        }, None).await.ok()?;

        let shader_source = std::fs::read_to_string("src/shader.wgsl")
            .unwrap_or_else(|_| include_str!("shader.wgsl").to_string());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source)),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        let fxaa_source = std::fs::read_to_string("src/fxaa.wgsl")
            .unwrap_or_else(|_| include_str!("fxaa.wgsl").to_string());

        let fxaa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(fxaa_source)),
        });

        let fxaa_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &fxaa_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        let restoration_source = std::fs::read_to_string("src/restoration.wgsl")
            .unwrap_or_else(|_| include_str!("restoration.wgsl").to_string());

        let restoration_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Restoration Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(restoration_source)),
        });

        let restoration_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Restoration Pipeline"),
            layout: None,
            module: &restoration_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        let refine_precompute_source = std::fs::read_to_string("src/refine_precompute.wgsl")
            .unwrap_or_else(|_| include_str!("refine_precompute.wgsl").to_string());

        let refine_precompute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Refinement Precompute Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(refine_precompute_source)),
        });

        let refine_precompute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Refinement Precompute Pipeline"),
            layout: None,
            module: &refine_precompute_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        Some(Self { device, queue, compute_pipeline, fxaa_pipeline, restoration_pipeline, refine_precompute_pipeline })
    }

    /// `origin_x`/`origin_y` are this tile's absolute OUTPUT-space offset in
    /// the final image (0, 0 if this call represents the whole image, i.e.
    /// no tiling). See the `Params` doc comment for why this exists.
    ///
    /// `use_precomputed_refinement` is the EXPERIMENTAL toggle described in
    /// shader.wgsl / refine_precompute.wgsl — off by default, safe to leave
    /// off, easy to delete (see those files' doc comments) if it doesn't
    /// pay off.
    #[allow(clippy::too_many_arguments)]
    pub fn upscale_tile(&self, input_data: &[f32], width: u32, height: u32, channels: u32, scale: f32, contrast_thresh: f32, blend_max: f32, refine: bool, algorithm: u32, operation_mode: u32, restore_filter: u32, bilateral_tol: f32, deblock_int: f32, origin_x: u32, origin_y: u32, use_precomputed_refinement: bool, bilateral_radius: u32) -> Vec<f32> {
        let out_width = (width as f32 * scale).ceil() as u32;
        let out_height = (height as f32 * scale).ceil() as u32;
        // Widen to u64 before multiplying: per-tile dims are gated by the
        // device's storage buffer binding limit elsewhere, but this stays
        // correct even if that invariant is ever relaxed.
        let out_size = (out_width as u64 * out_height as u64 * channels as u64) as usize;

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (out_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = Params {
            width, height, channels, scale, contrast_thresh, blend_max, algorithm,
            operation_mode, restore_filter, bilateral_tol, deblock_int,
            origin_x, origin_y,
            use_precomputed_refinement: use_precomputed_refinement as u32,
            bilateral_radius, _pad5: 0,
        };
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // EXPERIMENTAL precompute pass: once per INPUT pixel instead of
        // once per output pixel — see refine_precompute.wgsl. Runs (in the
        // same encoder, before the main pass, so wgpu orders the writes
        // before the read) only when the caller opts in; otherwise a tiny
        // 1-element dummy buffer is bound in its place so the main
        // pipeline's fixed 4-binding layout is still satisfied.
        let precompute_record_floats: u64 = width as u64 * height as u64 * 6;
        let precomputed_refinement_buffer = if use_precomputed_refinement && operation_mode != 1 {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Refinement Precompute Buffer"),
                size: precompute_record_floats * std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let precompute_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.refine_precompute_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                ],
            });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.refine_precompute_pipeline);
                cpass.set_bind_group(0, &precompute_bind_group, &[]);
                let workgroups_x = (width + 15) / 16;
                let workgroups_y = (height + 15) / 16;
                cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            buf
        } else {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Refinement Precompute Buffer (unused)"),
                size: std::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };

        let bind_group_layout = if operation_mode == 1 {
            self.restoration_pipeline.get_bind_group_layout(0)
        } else {
            self.compute_pipeline.get_bind_group_layout(0)
        };
        let bind_group = if operation_mode == 1 {
            // restoration.wgsl only declares 3 bindings (no
            // precomputed_refinement) — its bind group layout has no slot
            // for a 4th entry.
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                ],
            })
        } else {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: precomputed_refinement_buffer.as_entire_binding() },
                ],
            })
        };

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            if operation_mode == 1 {
                cpass.set_pipeline(&self.restoration_pipeline);
            } else {
                cpass.set_pipeline(&self.compute_pipeline);
            }
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = (out_width + 15) / 16;
            let workgroups_y = (out_height + 15) / 16;
            cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        let fxaa_output_buffer = if refine {
            Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("FXAA Output Buffer"),
                size: (out_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        let final_buffer = if refine {
            let fxaa_out = fxaa_output_buffer.as_ref().unwrap();
            let fxaa_params = Params {
                width: out_width, height: out_height, channels, scale: 1.0, contrast_thresh: 0.0, blend_max: 0.0, algorithm: 0,
                operation_mode: 0, restore_filter: 0, bilateral_tol: 0.0, deblock_int: 0.0,
                origin_x: 0, origin_y: 0, use_precomputed_refinement: 0, bilateral_radius: 0, _pad5: 0,
            };
            let fxaa_params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FXAA Params Buffer"),
                contents: bytemuck::bytes_of(&fxaa_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let fxaa_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.fxaa_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: output_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: fxaa_out.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: fxaa_params_buffer.as_entire_binding() },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.fxaa_pipeline);
                cpass.set_bind_group(0, &fxaa_bind_group, &[]);
                let workgroups_x = (out_width + 15) / 16;
                let workgroups_y = (out_height + 15) / 16;
                cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
            fxaa_out
        } else {
            &output_buffer
        };

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: final_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(final_buffer, 0, &staging_buffer, 0, final_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);

        if receiver.recv().unwrap().is_ok() {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            result
        } else {
            panic!("Failed to map GPU buffer");
        }
    }
}

#[cfg(test)]
mod tiling_equivalence_test {
    use super::GpuContext;

    /// Deterministic synthetic image mixing a smooth gradient (exercises the
    /// Lanczos path) with a coarse checkerboard and per-pixel noise
    /// (exercises the local-contrast "conservative auto-expanding gridding"
    /// refinement path in shader.wgsl, which is the part with the largest
    /// neighborhood search radius).
    fn make_synthetic_image(width: u32, height: u32) -> Vec<f32> {
        let mut data = vec![0.0f32; (width * height * 3) as usize];
        let mut state: u32 = 0x9E3779B9;
        let mut next_rand = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32) / (u32::MAX as f32)
        };
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                let gradient = x as f32 / width as f32;
                let checker = if ((x / 32) + (y / 32)) % 2 == 0 { 1.0 } else { 0.0 };
                let noise = next_rand() * 0.15;
                data[idx] = (gradient * 0.6 + checker * 0.3 + noise).clamp(0.0, 1.0);
                data[idx + 1] = ((1.0 - gradient) * 0.5 + checker * 0.2 + noise).clamp(0.0, 1.0);
                data[idx + 2] = ((y as f32 / height as f32) * 0.7 + noise).clamp(0.0, 1.0);
            }
        }
        data
    }

    /// Runs a full-image reference upscale next to a tiled/hard-cropped one
    /// (no feathering, no alpha blend) at the given scale, and returns
    /// `Err` describing the first mismatch if they are not bit-for-bit
    /// identical. `width`/`height` must be evenly divisible by
    /// `tiles_per_side`.
    ///
    /// This is the shared core behind every
    /// `tiled_hard_crop_matches_full_image_bit_for_bit_at_*` test below —
    /// one bug-for-bug faithful implementation, exercised at multiple scale
    /// factors, instead of copy-pasted per-scale test bodies that could
    /// silently drift apart.
    fn check_tiling_equivalence(
        gpu_ctx: &GpuContext,
        width: u32,
        height: u32,
        scale: f32,
        tiles_per_side: u32,
        padding: u32,
    ) -> Result<(), String> {
        let contrast_thresh = 0.25f32;
        let blend_max = 0.5f32;
        let algorithm = 1u32; // Lanczos3
        let operation_mode = 0u32;
        let restore_filter = 0u32;
        let bilateral_tol = 0.1f32;
        let deblock_int = 0.5f32;

        let image = make_synthetic_image(width, height);

        // Reference: whole image processed as a single tile, exactly like
        // an unlimited-VRAM, no-tiling run.
        let reference = gpu_ctx.upscale_tile(
            &image, width, height, 3, scale, contrast_thresh, blend_max, false, algorithm,
            operation_mode, restore_filter, bilateral_tol, deblock_int, 0, 0, false, 4,
        );

        // Candidate: same image split into a tile grid with `padding`
        // pixels of real context on each side, hard-cropped back to each
        // tile's core region (no feathering, no alpha blend) and
        // reassembled with a plain overwrite.
        let tile_w = width / tiles_per_side;
        let tile_h = height / tiles_per_side;
        let out_w = (width as f32 * scale).ceil() as u32;
        let out_h = (height as f32 * scale).ceil() as u32;
        let mut candidate = vec![0.0f32; (out_w as u64 * out_h as u64 * 3) as usize];

        for ty in 0..tiles_per_side {
            for tx in 0..tiles_per_side {
                let start_x = tx * tile_w;
                let start_y = ty * tile_h;
                let end_x = start_x + tile_w;
                let end_y = start_y + tile_h;

                let padded_start_x = start_x.saturating_sub(padding);
                let padded_start_y = start_y.saturating_sub(padding);
                let padded_end_x = (end_x + padding).min(width);
                let padded_end_y = (end_y + padding).min(height);
                let padded_w = padded_end_x - padded_start_x;
                let padded_h = padded_end_y - padded_start_y;

                let mut tile_data = vec![0.0f32; (padded_w * padded_h * 3) as usize];
                for y in 0..padded_h {
                    for x in 0..padded_w {
                        let src_idx =
                            (((padded_start_y + y) * width + (padded_start_x + x)) * 3) as usize;
                        let dst_idx = ((y * padded_w + x) * 3) as usize;
                        tile_data[dst_idx] = image[src_idx];
                        tile_data[dst_idx + 1] = image[src_idx + 1];
                        tile_data[dst_idx + 2] = image[src_idx + 2];
                    }
                }

                let out_padded_start_x = (padded_start_x as f32 * scale).ceil() as u32;
                let out_padded_start_y = (padded_start_y as f32 * scale).ceil() as u32;

                let upscaled = gpu_ctx.upscale_tile(
                    &tile_data, padded_w, padded_h, 3, scale, contrast_thresh, blend_max, false,
                    algorithm, operation_mode, restore_filter, bilateral_tol, deblock_int,
                    out_padded_start_x, out_padded_start_y, false, 4,
                );

                // Hard crop: discard the scaled padding border entirely,
                // keep only the core region that maps 1:1 to [start_x,end_x).
                let out_padded_w = (padded_w as f32 * scale).ceil() as u32;
                let crop_left = ((start_x - padded_start_x) as f32 * scale).round() as u32;
                let crop_top = ((start_y - padded_start_y) as f32 * scale).round() as u32;
                let core_out_w = ((end_x - start_x) as f32 * scale).round() as u32;
                let core_out_h = ((end_y - start_y) as f32 * scale).round() as u32;
                let out_start_x = (start_x as f32 * scale).round() as u32;
                let out_start_y = (start_y as f32 * scale).round() as u32;

                for y in 0..core_out_h {
                    for x in 0..core_out_w {
                        let src_idx =
                            (((crop_top + y) * out_padded_w + (crop_left + x)) * 3) as usize;
                        let dst_idx = (((out_start_y + y) as u64 * out_w as u64
                            + (out_start_x + x) as u64)
                            * 3) as usize;
                        candidate[dst_idx] = upscaled[src_idx];
                        candidate[dst_idx + 1] = upscaled[src_idx + 1];
                        candidate[dst_idx + 2] = upscaled[src_idx + 2];
                    }
                }
            }
        }

        if reference.len() != candidate.len() {
            return Err(format!(
                "buffer length mismatch: reference={}, candidate={}",
                reference.len(),
                candidate.len()
            ));
        }

        // Track both raw f32 mismatches AND whether the divergence survives
        // quantization to the u8 that actually ends up in the PNG — a
        // sub-ULP floating point rounding difference and a real structural
        // (insufficient padding radius) bug look identical as "N floats
        // differ" but have very different implications.
        let mut mismatch_count = 0usize;
        let mut u8_mismatch_count = 0usize;
        let mut max_abs_diff = 0.0f32;
        let mut max_abs_diff_idx = 0usize;
        let mut first_mismatch: Option<(usize, f32, f32)> = None;
        let mut first_u8_mismatch: Option<(usize, f32, f32, u8, u8)> = None;
        for i in 0..reference.len() {
            if reference[i] != candidate[i] {
                mismatch_count += 1;
                let diff = (reference[i] - candidate[i]).abs();
                if diff > max_abs_diff {
                    max_abs_diff = diff;
                    max_abs_diff_idx = i;
                }
                if first_mismatch.is_none() {
                    first_mismatch = Some((i, reference[i], candidate[i]));
                }
                let ref_u8 = (reference[i].clamp(0.0, 1.0) * 255.0).round() as u8;
                let cand_u8 = (candidate[i].clamp(0.0, 1.0) * 255.0).round() as u8;
                if ref_u8 != cand_u8 {
                    u8_mismatch_count += 1;
                    if first_u8_mismatch.is_none() {
                        first_u8_mismatch =
                            Some((i, reference[i], candidate[i], ref_u8, cand_u8));
                    }
                }
            }
        }

        if mismatch_count > 0 {
            return Err(format!(
                "scale={}: {}/{} f32 elements differ (max abs diff={:e} at index {}, ref={}, cand={}), of which {} survive u8 quantization to a different byte{}",
                scale,
                mismatch_count,
                reference.len(),
                max_abs_diff,
                max_abs_diff_idx,
                reference[max_abs_diff_idx],
                candidate[max_abs_diff_idx],
                u8_mismatch_count,
                first_u8_mismatch
                    .map(|(i, r, c, ru, cu)| format!(
                        " (first u8 mismatch at index {}: f32 ref={} cand={} -> u8 ref={} cand={})",
                        i, r, c, ru, cu
                    ))
                    .unwrap_or_else(|| first_mismatch
                        .map(|(i, r, c)| format!(
                            " (none; first f32-only mismatch at index {}: ref={} cand={})",
                            i, r, c
                        ))
                        .unwrap_or_default())
            ));
        }

        Ok(())
    }

    /// Requires a real GPU adapter, so these are `#[ignore]`d by default.
    /// Run explicitly with: `cargo test --release -- --ignored tiled_hard_crop`
    ///
    /// One GPU context is reused across all scale factors so the (slow,
    /// one-time) adapter/device setup only happens once per test binary run.
    #[test]
    #[ignore]
    fn tiled_hard_crop_matches_full_image_bit_for_bit_across_scales() {
        let gpu_ctx =
            pollster::block_on(GpuContext::new()).expect("GPU adapter required for this test");
        let padding: u32 = 16; // matches the production default

        // Original case: moderate scale, larger source. Exercises the
        // hypothesis at the scale factor most users will actually run.
        //
        // The scale factors that actually motivated this investigation
        // (50x / 100x on a small source, e.g. 512x512 -> 25600x25600 /
        // 51200x51200). Kept the SOURCE small here (128x128) so the test
        // still runs in seconds — what matters for this hypothesis is the
        // SCALE FACTOR (it changes how far `fx = out_x / scale` can land
        // from the tile's real pixel grid), not the final output size.
        let cases: &[(u32, u32, f32)] = &[(2048, 2048, 4.0), (128, 128, 50.0), (128, 128, 100.0)];
        let mut failures = Vec::new();
        for &(w, h, scale) in cases {
            match check_tiling_equivalence(&gpu_ctx, w, h, scale, 4, padding) {
                Ok(()) => println!("OK: {}x{} @ {}x — bit-for-bit identical", w, h, scale),
                Err(msg) => {
                    println!("MISMATCH: {}x{} @ {}x — {}", w, h, scale, msg);
                    failures.push(msg);
                }
            }
        }

        if !failures.is_empty() {
            panic!("{} of {} scale cases diverged:\n{}", failures.len(), cases.len(), failures.join("\n"));
        }
    }

    /// Diagnostic (not a pass/fail gate): if the scale=50/100 mismatches
    /// were caused by insufficient neighborhood context, a much larger
    /// padding should shrink or eliminate them. If the mismatch count stays
    /// essentially flat as padding grows, the cause is something else
    /// (coordinate-magnitude-dependent float rounding, not context radius).
    #[test]
    #[ignore]
    fn padding_sensitivity_probe_at_scale_50() {
        let gpu_ctx =
            pollster::block_on(GpuContext::new()).expect("GPU adapter required for this test");
        for &padding in &[16u32, 32, 64] {
            match check_tiling_equivalence(&gpu_ctx, 128, 128, 50.0, 4, padding) {
                Ok(()) => println!("padding={}: OK, bit-for-bit identical", padding),
                Err(msg) => println!("padding={}: {}", padding, msg),
            }
        }
    }

    /// Flat-color image with a single hard vertical edge down the middle —
    /// much closer to actual anime/flat-art content than
    /// `make_synthetic_image`'s deliberate worst-case stress pattern
    /// (checkerboard + gradient + noise everywhere). Used to check whether
    /// a large measured divergence is a real bug or just an artifact of
    /// stress-testing with near-constant contrast across the whole image.
    fn make_flat_color_with_edge(width: u32, height: u32) -> Vec<f32> {
        let mut data = vec![0.0f32; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                let left = x < width / 2;
                let (r, g, b) = if left { (0.85, 0.2, 0.2) } else { (0.15, 0.6, 0.9) };
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
            }
        }
        data
    }

    #[test]
    #[ignore]
    fn measure_precomputed_refinement_divergence_on_flat_art() {
        let gpu_ctx =
            pollster::block_on(GpuContext::new()).expect("GPU adapter required for this test");

        let (width, height) = (256u32, 256u32);
        let scale = 8.0f32;
        let image = make_flat_color_with_edge(width, height);

        let baseline = gpu_ctx.upscale_tile(
            &image, width, height, 3, scale, 0.25, 0.5, false, 1, 0, 0, 0.1, 0.5, 0, 0, false, 4,
        );
        let precomputed = gpu_ctx.upscale_tile(
            &image, width, height, 3, scale, 0.25, 0.5, false, 1, 0, 0, 0.1, 0.5, 0, 0, true, 4,
        );

        assert_eq!(baseline.len(), precomputed.len());

        let mut mismatch_count = 0usize;
        let mut u8_mismatch_count = 0usize;
        let mut max_abs_diff = 0.0f32;
        for i in 0..baseline.len() {
            let diff = (baseline[i] - precomputed[i]).abs();
            if diff > 0.0 {
                mismatch_count += 1;
                if diff > max_abs_diff {
                    max_abs_diff = diff;
                }
                let base_u8 = (baseline[i].clamp(0.0, 1.0) * 255.0).round() as u8;
                let pre_u8 = (precomputed[i].clamp(0.0, 1.0) * 255.0).round() as u8;
                if base_u8 != pre_u8 {
                    u8_mismatch_count += 1;
                }
            }
        }

        println!(
            "precomputed refinement vs baseline (flat-color+edge) at {}x{} @ {}x: {}/{} f32 elements differ ({:.4}%), max abs diff={:e}, {}/{} survive u8 quantization ({:.4}%)",
            width, height, scale,
            mismatch_count, baseline.len(), 100.0 * mismatch_count as f64 / baseline.len() as f64,
            max_abs_diff,
            u8_mismatch_count, baseline.len(), 100.0 * u8_mismatch_count as f64 / baseline.len() as f64,
        );
    }

    /// EXPERIMENTAL `use_precomputed_refinement` is a real approximation,
    /// not a bit-exact refactor (see the doc comments in shader.wgsl and
    /// refine_precompute.wgsl for why: the precompute pass has no
    /// fractional output position to decide `target_l` with, so it uses
    /// this pixel's own raw value instead of the interpolated base_color).
    /// This is a diagnostic, not a pass/fail gate — it quantifies how much
    /// the optimization changes the output on a high-contrast synthetic
    /// image, so that decision is informed by numbers instead of a guess.
    #[test]
    #[ignore]
    fn measure_precomputed_refinement_divergence() {
        let gpu_ctx =
            pollster::block_on(GpuContext::new()).expect("GPU adapter required for this test");

        let (width, height) = (256u32, 256u32);
        let scale = 8.0f32;
        let image = make_synthetic_image(width, height);

        let baseline = gpu_ctx.upscale_tile(
            &image, width, height, 3, scale, 0.25, 0.5, false, 1, 0, 0, 0.1, 0.5, 0, 0, false, 4,
        );
        let precomputed = gpu_ctx.upscale_tile(
            &image, width, height, 3, scale, 0.25, 0.5, false, 1, 0, 0, 0.1, 0.5, 0, 0, true, 4,
        );

        assert_eq!(baseline.len(), precomputed.len());

        let mut mismatch_count = 0usize;
        let mut u8_mismatch_count = 0usize;
        let mut max_abs_diff = 0.0f32;
        for i in 0..baseline.len() {
            let diff = (baseline[i] - precomputed[i]).abs();
            if diff > 0.0 {
                mismatch_count += 1;
                if diff > max_abs_diff {
                    max_abs_diff = diff;
                }
                let base_u8 = (baseline[i].clamp(0.0, 1.0) * 255.0).round() as u8;
                let pre_u8 = (precomputed[i].clamp(0.0, 1.0) * 255.0).round() as u8;
                if base_u8 != pre_u8 {
                    u8_mismatch_count += 1;
                }
            }
        }

        println!(
            "precomputed refinement vs baseline at {}x{} @ {}x: {}/{} f32 elements differ ({:.4}%), max abs diff={:e}, {}/{} survive u8 quantization ({:.4}%)",
            width, height, scale,
            mismatch_count, baseline.len(), 100.0 * mismatch_count as f64 / baseline.len() as f64,
            max_abs_diff,
            u8_mismatch_count, baseline.len(), 100.0 * u8_mismatch_count as f64 / baseline.len() as f64,
        );
    }

    /// One-off manual visual check (not a pass/fail gate, deliberately not
    /// named to match test-runner conventions users would rely on) — runs
    /// the real bilateral restoration filter against a real source image on
    /// disk and writes the result next to it, to compare the widened
    /// 9x9-window kernel against a real photo instead of reasoning about it
    /// only in the abstract. Ignored by default; the path is a local temp
    /// file that won't exist on other machines.
    #[test]
    #[ignore]
    fn render_bilateral_restoration_preview() {
        let path = std::path::Path::new(
            "C:/Users/leonp/AppData/Local/Temp/claude/C--Users-leonp-OneDrive-Escritorio-Upscaller/a6288e0c-824c-4e30-80a5-19bb1742cab5/images/3.jpg",
        );
        if !path.exists() {
            println!("skipping: source image not present at {:?}", path);
            return;
        }
        let gpu_ctx =
            pollster::block_on(GpuContext::new()).expect("GPU adapter required for this test");

        let img = image::open(path).expect("failed to load source image").to_rgb8();
        let (width, height) = (img.width(), img.height());
        let mut data = vec![0.0f32; (width * height * 3) as usize];
        for (i, p) in img.pixels().enumerate() {
            data[i * 3] = p[0] as f32 / 255.0;
            data[i * 3 + 1] = p[1] as f32 / 255.0;
            data[i * 3 + 2] = p[2] as f32 / 255.0;
        }

        for radius in [2u32, 4, 8] {
            let result = gpu_ctx.upscale_tile(
                &data, width, height, 3, 1.0, 0.25, 0.5, false, 1, 1, 0, 0.1, 0.5, 0, 0, false, radius,
            );

            let mut out_img = image::RgbImage::new(width, height);
            for y in 0..height {
                for x in 0..width {
                    let idx = ((y * width + x) * 3) as usize;
                    out_img.put_pixel(x, y, image::Rgb([
                        (result[idx].clamp(0.0, 1.0) * 255.0) as u8,
                        (result[idx + 1].clamp(0.0, 1.0) * 255.0) as u8,
                        (result[idx + 2].clamp(0.0, 1.0) * 255.0) as u8,
                    ]));
                }
            }
            let out_path = path.with_file_name(format!("bilateral_r{}_preview.png", radius));
            out_img.save(&out_path).expect("failed to save preview");
            println!("wrote {:?}", out_path);
        }
    }

    /// Renders a REAL upscaled (Lanczos + adaptive gridding, the actual
    /// `Escalar Resolución` pipeline — not restoration/bilateral) image at
    /// scale=4x from the real source photo, so downstream staircase-fix
    /// prototyping happens on genuine upscale output (with real "room"
    /// between original source pixels) instead of a native-resolution
    /// restoration pass, which was the methodological mistake in earlier
    /// prototyping this session.
    #[test]
    #[ignore]
    fn render_real_upscale_preview() {
        let path = std::path::Path::new(
            "C:/Users/leonp/AppData/Local/Temp/claude/C--Users-leonp-OneDrive-Escritorio-Upscaller/a6288e0c-824c-4e30-80a5-19bb1742cab5/images/3.jpg",
        );
        if !path.exists() {
            println!("skipping: source image not present at {:?}", path);
            return;
        }
        let gpu_ctx =
            pollster::block_on(GpuContext::new()).expect("GPU adapter required for this test");

        let img = image::open(path).expect("failed to load source image").to_rgb8();
        let (width, height) = (img.width(), img.height());
        let mut data = vec![0.0f32; (width * height * 3) as usize];
        for (i, p) in img.pixels().enumerate() {
            data[i * 3] = p[0] as f32 / 255.0;
            data[i * 3 + 1] = p[1] as f32 / 255.0;
            data[i * 3 + 2] = p[2] as f32 / 255.0;
        }

        let scale = 8.0f32;
        for blend_max in [0.5f32, 0.0f32] {
            let result = gpu_ctx.upscale_tile(
                &data, width, height, 3, scale, 0.25, blend_max, false, 1, 0, 0, 0.1, 0.5, 0, 0, false, 4,
            );

            let out_w = (width as f32 * scale).ceil() as u32;
            let out_h = (height as f32 * scale).ceil() as u32;
            let mut out_img = image::RgbImage::new(out_w, out_h);
            for y in 0..out_h {
                for x in 0..out_w {
                    let idx = ((y * out_w + x) * 3) as usize;
                    out_img.put_pixel(x, y, image::Rgb([
                        (result[idx].clamp(0.0, 1.0) * 255.0) as u8,
                        (result[idx + 1].clamp(0.0, 1.0) * 255.0) as u8,
                        (result[idx + 2].clamp(0.0, 1.0) * 255.0) as u8,
                    ]));
                }
            }
            let out_path = path.with_file_name(format!("real_upscale_4x_blend{}_preview.png", blend_max));
            out_img.save(&out_path).expect("failed to save preview");
            println!("wrote {:?}", out_path);
        }
    }
}
