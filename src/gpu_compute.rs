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
    pub _pad2: u32,
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub compute_pipeline: wgpu::ComputePipeline,
    pub fxaa_pipeline: wgpu::ComputePipeline,
    pub restoration_pipeline: wgpu::ComputePipeline,
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

        Some(Self { device, queue, compute_pipeline, fxaa_pipeline, restoration_pipeline })
    }

    pub fn upscale_tile(&self, input_data: &[f32], width: u32, height: u32, channels: u32, scale: f32, contrast_thresh: f32, blend_max: f32, refine: bool, algorithm: u32, operation_mode: u32, restore_filter: u32, bilateral_tol: f32, deblock_int: f32) -> Vec<f32> {
        let out_width = (width as f32 * scale).ceil() as u32;
        let out_height = (height as f32 * scale).ceil() as u32;
        let out_size = (out_width * out_height * channels) as usize;

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
            operation_mode, restore_filter, bilateral_tol, deblock_int, _pad2: 0 
        };
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group_layout = if operation_mode == 1 {
            self.restoration_pipeline.get_bind_group_layout(0)
        } else {
            self.compute_pipeline.get_bind_group_layout(0)
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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
                operation_mode: 0, restore_filter: 0, bilateral_tol: 0.0, deblock_int: 0.0, _pad2: 0 
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
