//! GPU 初始化与 3D 体素渲染。
//!
//! 这一层只负责资源和 draw call；世界查询、面剔除和网格生命周期由上层处理。

use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use winit::dpi::PhysicalSize;
use winit::window::Window;

use super::voxel::{Vertex, VoxelMesh};
use crate::world::atlas::{self, Atlas};

const MENU_CLEAR: wgpu::Color = wgpu::Color {
    r: 0.05,
    g: 0.07,
    b: 0.12,
    a: 1.0,
};

pub const WORLD_CLEAR: wgpu::Color = wgpu::Color {
    r: 0.38,
    g: 0.62,
    b: 0.84,
    a: 1.0,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    view_proj: [[f32; 4]; 4],
    /// 从被照表面指向太阳的方向。
    sun_direction: [f32; 4],
    /// RGB 为太阳颜色，A 为太阳强度。
    sun_color: [f32; 4],
    /// RGB 为天空环境光颜色，A 为环境光强度。
    sky_color: [f32; 4],
    /// 游戏运行时间（秒），供水面波纹使用。
    time: [f32; 4],
}

const VOXEL_SHADER: &str = r#"
struct Frame {
    view_proj: mat4x4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    time: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> frame: Frame;

@group(0) @binding(1)
var atlas_texture: texture_2d<f32>;

@group(0) @binding(2)
var atlas_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) ao: f32,
    @location(4) material: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) ao: f32,
    @location(3) material: f32,
    @location(4) world_position: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    var position = input.position;
    let is_water = input.material > 0.5;
    let top_wave = sin(position.x * 2.2 + frame.time.x * 1.15)
        * cos(position.z * 1.7 + frame.time.x * 0.85) * 0.028;
    if is_water && input.normal.y > 0.5 {
        position.y = position.y + top_wave;
    }
    output.clip_position = frame.view_proj * vec4<f32>(position, 1.0);
    output.uv = input.uv;
    output.normal = input.normal;
    output.ao = input.ao;
    output.material = input.material;
    output.world_position = position;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas_texture, atlas_sampler, input.uv);
    // Transparent texture pixels must not write depth. Glass is represented by
    // an opaque detail pattern on a transparent background; without this test,
    // the alpha-blended color is invisible but the depth value still hides the
    // terrain behind it, producing large clear-colored planes.
    if texel.a < 0.5 {
        discard;
    }
    let is_water = input.material > 0.5;
    let underwater = frame.time.y > 0.5;
    var color = texel.rgb;
    var alpha = texel.a;
    var normal = input.normal;
    if is_water {
        let wave_a = sin(input.world_position.x * 2.2 + frame.time.x * 1.15)
            * cos(input.world_position.z * 1.7 + frame.time.x * 0.85);
        let wave_b = sin((input.world_position.x + input.world_position.z) * 4.0
            - frame.time.x * 0.7);
        let ripple = 0.5 + 0.5 * (wave_a * 0.7 + wave_b * 0.3);
        let deep = vec3<f32>(0.025, 0.27, 0.42);
        let sunlit = vec3<f32>(0.10, 0.58, 0.68);
        color = mix(deep, sunlit, 0.35 + ripple * 0.35);
        if input.normal.y > 0.5 {
            normal = normalize(input.normal + vec3<f32>(
                wave_b * 0.12,
                0.0,
                wave_a * 0.12,
            ));
        }
        // 让水下方块保持可读，同时保留明显的蓝绿色玻璃感。
        alpha = 0.54 + ripple * 0.10;
        color = color + vec3<f32>(0.08, 0.14, 0.13) * max(wave_b, 0.0);
    }
    if underwater && !is_water {
        let distance = length(input.world_position);
        let fog = clamp((distance - 2.5) / 28.0, 0.0, 1.0) * 0.72;
        let water_fog = vec3<f32>(0.035, 0.34, 0.40);
        color = mix(color, water_fog, fog);
        let caustic = max(
            sin(input.world_position.x * 2.1 + frame.time.x * 1.2)
                * sin(input.world_position.z * 1.7 - frame.time.x * 0.9),
            0.0,
        );
        color = color + vec3<f32>(0.05, 0.12, 0.10) * caustic * (1.0 - fog) * 0.8;
    }
    normal = normalize(normal);
    let sun_direction = normalize(frame.sun_direction.xyz);
    let direct = max(dot(normal, sun_direction), 0.0)
        * frame.sun_color.a
        * frame.sun_color.rgb;
    var ambient_color = frame.sky_color.rgb;
    if underwater {
        ambient_color = vec3<f32>(0.04, 0.30, 0.36);
    }
    let ambient = ambient_color * frame.sky_color.a;
    let lighting = (ambient + direct) * input.ao;
    return vec4<f32>(color * lighting, alpha);
}
"#;

const HIGHLIGHT_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> @builtin(position) vec4<f32> {
    return camera.view_proj * vec4<f32>(input.position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.02, 0.02, 0.02, 0.9);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HighlightVertex {
    position: [f32; 3],
}

impl HighlightVertex {
    const fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
        }
    }
}

pub struct GpuState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub atlas: Atlas,
    depth_view: wgpu::TextureView,
    voxel_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    highlight_pipeline: wgpu::RenderPipeline,
    voxel_bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    vertex_capacity: u64,
    index_capacity: u64,
    index_count: u32,
    transparent_vertex_buffer: wgpu::Buffer,
    transparent_index_buffer: wgpu::Buffer,
    transparent_vertex_capacity: u64,
    transparent_index_capacity: u64,
    transparent_index_count: u32,
    highlight_buffer: wgpu::Buffer,
    highlight_capacity: u64,
    highlight_count: u32,
    start_time: Instant,
    /// `Surface<'static>` 要求窗口句柄一直有效，`Arc` 保活。
    #[allow(dead_code)]
    window: Arc<Window>,
}

impl GpuState {
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size().max(PhysicalSize::new(1, 1));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("创建 Surface 失败");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("找不到可用 GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mc device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .expect("创建 Device 失败");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let atlas = atlas::build(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("assets/textures")
                .as_path(),
        )
        .expect("构建体素纹理 atlas 失败");
        let depth_view = create_depth_view(&device, size);
        let (
            voxel_pipeline,
            transparent_pipeline,
            highlight_pipeline,
            voxel_bind_group,
            camera_buffer,
        ) = create_voxel_pipeline(&device, &queue, &config, &atlas);
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc voxel vertices"),
            size: std::mem::size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc voxel indices"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transparent_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc transparent voxel vertices"),
            size: std::mem::size_of::<Vertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transparent_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc transparent voxel indices"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let highlight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc block highlight"),
            size: std::mem::size_of::<HighlightVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            surface,
            config,
            atlas,
            depth_view,
            voxel_pipeline,
            transparent_pipeline,
            highlight_pipeline,
            voxel_bind_group,
            camera_buffer,
            vertex_buffer,
            index_buffer,
            index_count: 0,
            vertex_capacity: std::mem::size_of::<Vertex>() as u64,
            index_capacity: std::mem::size_of::<u32>() as u64,
            highlight_buffer,
            highlight_capacity: std::mem::size_of::<HighlightVertex>() as u64,
            highlight_count: 0,
            transparent_vertex_buffer,
            transparent_index_buffer,
            transparent_vertex_capacity: std::mem::size_of::<Vertex>() as u64,
            transparent_index_capacity: std::mem::size_of::<u32>() as u64,
            transparent_index_count: 0,
            start_time: Instant::now(),
            window,
        }
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth_view(&self.device, size);
    }

    pub fn upload_mesh(&mut self, mesh: &VoxelMesh) {
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            self.index_count = 0;
        } else {
            let vertex_bytes = bytemuck::cast_slice(&mesh.vertices);
            let index_bytes = bytemuck::cast_slice(&mesh.indices);
            if vertex_bytes.len() as u64 > self.vertex_capacity {
                self.vertex_capacity = vertex_bytes.len().next_power_of_two() as u64;
                self.vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mc voxel vertices"),
                    size: self.vertex_capacity,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            if index_bytes.len() as u64 > self.index_capacity {
                self.index_capacity = index_bytes.len().next_power_of_two() as u64;
                self.index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mc voxel indices"),
                    size: self.index_capacity,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            self.queue
                .write_buffer(&self.vertex_buffer, 0, vertex_bytes);
            self.queue.write_buffer(&self.index_buffer, 0, index_bytes);
            self.index_count = mesh.indices.len() as u32;
        }

        if mesh.transparent_vertices.is_empty() || mesh.transparent_indices.is_empty() {
            self.transparent_index_count = 0;
        } else {
            let vertex_bytes = bytemuck::cast_slice(&mesh.transparent_vertices);
            let index_bytes = bytemuck::cast_slice(&mesh.transparent_indices);
            if vertex_bytes.len() as u64 > self.transparent_vertex_capacity {
                self.transparent_vertex_capacity = vertex_bytes.len().next_power_of_two() as u64;
                self.transparent_vertex_buffer =
                    self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("mc transparent voxel vertices"),
                        size: self.transparent_vertex_capacity,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
            }
            if index_bytes.len() as u64 > self.transparent_index_capacity {
                self.transparent_index_capacity = index_bytes.len().next_power_of_two() as u64;
                self.transparent_index_buffer =
                    self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("mc transparent voxel indices"),
                        size: self.transparent_index_capacity,
                        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
            }
            self.queue
                .write_buffer(&self.transparent_vertex_buffer, 0, vertex_bytes);
            self.queue
                .write_buffer(&self.transparent_index_buffer, 0, index_bytes);
            self.transparent_index_count = mesh.transparent_indices.len() as u32;
        }
    }

    pub fn update_camera(&self, view_proj: Mat4, underwater: bool) {
        // 当前先使用固定的白昼太阳；动态天空会继续复用这个 uniform。
        let sun_direction = glam::Vec3::new(0.55, 0.82, 0.35).normalize();
        let uniform = FrameUniform {
            view_proj: view_proj.to_cols_array_2d(),
            sun_direction: [sun_direction.x, sun_direction.y, sun_direction.z, 1.0],
            sun_color: [1.0, 0.93, 0.80, 0.90],
            sky_color: [0.48, 0.58, 0.72, 0.48],
            time: [
                self.start_time.elapsed().as_secs_f32(),
                underwater as u8 as f32,
                0.0,
                0.0,
            ],
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    pub fn draw_voxels<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.index_count == 0 {
            return;
        }
        pass.set_pipeline(&self.voxel_pipeline);
        pass.set_bind_group(0, &self.voxel_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    pub fn draw_transparent_voxels<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.transparent_index_count == 0 {
            return;
        }
        pass.set_pipeline(&self.transparent_pipeline);
        pass.set_bind_group(0, &self.voxel_bind_group, &[]);
        pass.set_vertex_buffer(0, self.transparent_vertex_buffer.slice(..));
        pass.set_index_buffer(
            self.transparent_index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        pass.draw_indexed(0..self.transparent_index_count, 0, 0..1);
    }

    pub fn upload_highlight(&mut self, block: Option<(i64, i64, i64)>, origin: glam::DVec3) {
        let Some((x, y, z)) = block else {
            self.highlight_count = 0;
            return;
        };
        let min =
            glam::DVec3::new(x as f64, y as f64, z as f64) - origin - glam::DVec3::splat(0.002);
        let max = min + glam::DVec3::splat(1.004);
        let corners = [
            [min.x as f32, min.y as f32, min.z as f32],
            [max.x as f32, min.y as f32, min.z as f32],
            [max.x as f32, min.y as f32, max.z as f32],
            [min.x as f32, min.y as f32, max.z as f32],
            [min.x as f32, max.y as f32, min.z as f32],
            [max.x as f32, max.y as f32, min.z as f32],
            [max.x as f32, max.y as f32, max.z as f32],
            [min.x as f32, max.y as f32, max.z as f32],
        ];
        let edge_indices = [
            0, 1, 1, 2, 2, 3, 3, 0, // bottom
            4, 5, 5, 6, 6, 7, 7, 4, // top
            0, 4, 1, 5, 2, 6, 3, 7, // sides
        ];
        let vertices: Vec<_> = edge_indices
            .into_iter()
            .map(|index| HighlightVertex {
                position: corners[index],
            })
            .collect();
        let bytes = bytemuck::cast_slice(&vertices);
        if bytes.len() as u64 > self.highlight_capacity {
            self.highlight_capacity = bytes.len().next_power_of_two() as u64;
            self.highlight_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc block highlight"),
                size: self.highlight_capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        self.queue.write_buffer(&self.highlight_buffer, 0, bytes);
        self.highlight_count = vertices.len() as u32;
    }

    pub fn draw_highlight<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.highlight_count == 0 {
            return;
        }
        pass.set_pipeline(&self.highlight_pipeline);
        pass.set_bind_group(0, &self.voxel_bind_group, &[]);
        pass.set_vertex_buffer(0, self.highlight_buffer.slice(..));
        pass.draw(0..self.highlight_count, 0..1);
    }

    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// 保留给无世界状态的简单清屏路径。
    pub fn render(&mut self) -> bool {
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return true;
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                let size = PhysicalSize::new(self.config.width, self.config.height);
                self.resize(size);
                return true;
            }
            wgpu::CurrentSurfaceTexture::Validation => return true,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mc clear frame"),
            });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mc clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(MENU_CLEAR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(output);
        true
    }
}

fn create_depth_view(device: &wgpu::Device, size: PhysicalSize<u32>) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("mc depth"),
            size: wgpu::Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_voxel_pipeline(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &wgpu::SurfaceConfiguration,
    atlas: &Atlas,
) -> (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::BindGroup,
    wgpu::Buffer,
) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mc voxel shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(VOXEL_SHADER)),
    });
    let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mc camera uniform"),
        size: std::mem::size_of::<FrameUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mc block atlas"),
        size: wgpu::Extent3d {
            width: atlas.size,
            height: atlas.size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        atlas.image.as_raw(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * atlas.size),
            rows_per_image: Some(atlas.size),
        },
        wgpu::Extent3d {
            width: atlas.size,
            height: atlas.size,
            depth_or_array_layers: 1,
        },
    );
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("mc block sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mc voxel bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mc voxel bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mc voxel pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let create_voxel_render_pipeline =
        |label: &'static str, depth_write_enabled: bool, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(Vertex::desc())],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(depth_write_enabled),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
    let pipeline = create_voxel_render_pipeline("mc voxel pipeline", true, None);
    let transparent_pipeline = create_voxel_render_pipeline(
        "mc transparent voxel pipeline",
        false,
        Some(wgpu::BlendState::ALPHA_BLENDING),
    );

    let highlight_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mc block highlight shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(HIGHLIGHT_SHADER)),
    });
    let highlight_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mc block highlight pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &highlight_shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(HighlightVertex::desc())],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            // wgpu 不允许 LineList 使用三角形专用的 depth bias。
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &highlight_shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    (
        pipeline,
        transparent_pipeline,
        highlight_pipeline,
        bind_group,
        camera_buffer,
    )
}
