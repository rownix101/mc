use super::*;

#[test]
fn shadow_camera_handles_zenith_horizon_and_night() {
    for sun in [glam::Vec3::Y, -glam::Vec3::Y, glam::Vec3::X] {
        let camera = glam::Vec3::new(32.0, 80.0, -16.0);
        let matrix = shadow_matrix(camera, sun);
        assert!(matrix.is_finite());
        let receiver = matrix.project_point3(camera);
        let caster = matrix.project_point3(camera + sun * 10.0);
        assert!(receiver.x.abs() < 0.001 && receiver.y.abs() < 0.001);
        assert!((0.0..1.0).contains(&receiver.z));
        assert!(caster.z < receiver.z, "sun-facing occluder must be nearer");
    }
}

#[test]
fn shadow_camera_snaps_subtexel_motion() {
    let a = shadow_matrix(glam::Vec3::ZERO, glam::Vec3::Y);
    let b = shadow_matrix(glam::Vec3::X * 0.001, glam::Vec3::Y);
    assert!(a.abs_diff_eq(b, 0.000001));
}

#[test]
fn lighting_pipelines_validate_on_gpu() {
    render_shader(VOXEL_SHADER, false, false, false);
}

#[test]
fn sand_sampling_is_continuous_across_patch_boundaries_and_not_periodic() {
    // Exercise the sand path with the fixture's textured atlas tile. Fix the
    // world position on either side of a patch boundary to isolate continuity
    // from geometry, lighting and rasterization changes.
    let sample = |x: f32| {
        let shader = VOXEL_SHADER
            .replace(
                "output.material = input.material;",
                "output.material = 4.0;",
            )
            .replace(
                "output.world_position = position + frame.world_origin.xyz;",
                &format!("output.world_position = vec3<f32>({x}, -0.37, 0.5);"),
            );
        render_shader(&shader, false, false, false)
    };
    let (Some(left), Some(right), Some(repeated)) =
        (sample(-0.00001), sample(0.00001), sample(2.00001))
    else {
        return;
    };
    assert!(
        left.iter().zip(&right).all(|(a, b)| a.abs_diff(*b) <= 1),
        "crossing a patch boundary must not introduce a visible jump"
    );
    assert_ne!(right, repeated, "sand must not repeat every two blocks");
}

#[test]
fn sky_fill_lifts_backlit_surfaces_but_preserves_enclosure_and_ao() {
    // Remove direct sun to measure indirect light on the side-facing test mesh.
    let indirect = VOXEL_SHADER.replace(
        "let direct = max(dot(normal, sun_direction), 0.0)",
        "let direct = 0.0 * max(dot(normal, sun_direction), 0.0)",
    );
    let baseline = indirect.replace(" + sky_fill + ground_bounce", "");
    let energy = |source: &str| {
        render_shader(source, false, false, false).map(|pixels| {
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| p[..3].iter().map(|&v| v as u64).sum::<u64>())
                .sum::<u64>()
        })
    };
    let (Some(before), Some(after)) = (energy(&baseline), energy(&indirect)) else {
        return;
    };
    assert!(after > before, "backlit material should receive sky fill");
    let sealed = |s: &str| s.replace("output.sky_light = input.light;", "output.sky_light = 0.0;");
    assert_eq!(
        energy(&sealed(&baseline)),
        energy(&sealed(&indirect)),
        "sky fill must not brighten fully enclosed geometry"
    );
    let occluded = indirect.replace("output.ao = input.ao;", "output.ao = input.ao * 0.35;");
    assert!(
        energy(&occluded).unwrap() < after,
        "contact AO must still darken indirect light"
    );
}

#[test]
fn optimized_shader_preserves_material_blends_and_water() {
    let reference = include_str!("testdata/voxel_reference.wgsl");
    for (water, blend) in [(false, false), (false, true), (true, false)] {
        let before = render_shader(reference, water, blend, false);
        let after = render_shader(VOXEL_SHADER, water, blend, false);
        if let (Some(before), Some(after)) = (before, after) {
            let max_error = before
                .iter()
                .zip(&after)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            eprintln!("water={water}, blend={blend}: maximum channel difference {max_error}/255");
            assert!(
                max_error <= 2,
                "water={water}, blend={blend}, max channel error={max_error}"
            );
        }
    }
}

#[test]
#[ignore = "GPU timestamp benchmark: 512x512 material and water fragment workload"]
fn fragment_gpu_benchmark() {
    for (water, blend) in [(false, true), (true, false)] {
        for (label, shader) in [
            ("reference", include_str!("testdata/voxel_reference.wgsl")),
            ("optimized", VOXEL_SHADER),
        ] {
            eprintln!("{label}, water={water}, blend={blend}");
            render_shader(shader, water, blend, true);
        }
    }
}

fn render_shader(
    source: &str,
    water_surface: bool,
    blend_edges: bool,
    benchmark: bool,
) -> Option<Vec<u8>> {
    render_shader_with_sky(source, water_surface, blend_edges, benchmark, None)
}

fn render_shader_with_sky(
    source: &str,
    water_surface: bool,
    blend_edges: bool,
    benchmark: bool,
    sky_frame: Option<FrameUniform>,
) -> Option<Vec<u8>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    else {
        eprintln!("Skipping GPU validation: no adapter available");
        return None;
    };
    if benchmark && !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        eprintln!("Skipping benchmark: adapter has no timestamp queries");
        return None;
    }
    if benchmark {
        eprintln!("adapter: {:?}", adapter.get_info());
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: if benchmark {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        },
        ..Default::default()
    }))
    .expect("test device");
    let dimension = if benchmark || sky_frame.is_some() {
        512
    } else {
        32
    };
    let row_bytes = (dimension * 4u32).div_ceil(256) * 256;
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        color_space: wgpu::SurfaceColorSpace::Auto,
        width: dimension,
        height: dimension,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    let atlas = atlas::build(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/textures")
            .as_path(),
    )
    .unwrap();
    let layout = create_scene_bind_group_layout(&device);
    let (_probe_texture, probe_view, probe_sampler) = create_probe_volume(&device);
    let (sky, voxel, water, _, bind, camera, shadow, shadow_view, shadow_bind) =
        create_voxel_pipeline_with_shader(
            &device,
            &queue,
            &config,
            &atlas,
            &layout,
            &ProbeBindings {
                view: &probe_view,
                sampler: &probe_sampler,
            },
            source,
        );
    let _composite = create_composite_pipeline(&device, &config, &layout);
    let godray_frame_layout = create_godray_frame_bind_group_layout(&device);
    let godray = create_godray_pipeline(&device, &config, &godray_frame_layout, &layout);
    let godray_frame_bind = create_godray_frame_bind_group(&device, &godray_frame_layout, &camera);
    let size = PhysicalSize::new(dimension, dimension);
    let target_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("culling pixel comparison"),
        size: wgpu::Extent3d {
            width: dimension,
            height: dimension,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target = target_texture.create_view(&Default::default());
    let depth = create_depth_view(&device, size);
    let (_, backdrop) = create_dummy_scene_texture(&device, &queue);
    let scene_bind = create_scene_bind_group(&device, &layout, &backdrop);
    let mut frame = FrameUniform::zeroed();
    frame.view_proj = Mat4::IDENTITY.to_cols_array_2d();
    frame.inv_view_proj = Mat4::IDENTITY.to_cols_array_2d();
    frame.shadow_view_proj = Mat4::IDENTITY.to_cols_array_2d();
    frame.time = [3.7, 0.0, 0.0, 0.0];
    frame.sun_direction = [0.0, 0.0, 1.0, 0.0];
    frame.camera_position = [0.0, 0.0, 2.0, 0.0];
    frame.viewport = [dimension as f32, dimension as f32, 0.0, 0.0];
    frame.sky_top = [0.2, 0.3, 0.4, 0.3];
    frame.sky_horizon = [0.4, 0.5, 0.6, 0.3];
    frame.sun_color = [1.0, 0.95, 0.8, 1.0];
    frame.params = [
        1.0,
        atlas.cols as f32,
        atlas::STRIDE as f32 / atlas.size as f32,
        atlas::GUTTER as f32 / atlas.size as f32,
    ];
    if let Some(preview) = sky_frame {
        frame = preview;
    }
    queue.write_buffer(&camera, 0, bytemuck::bytes_of(&frame));
    let stone = crate::world::block::Block::Stone;
    let tile = atlas.tile_id(stone, stone.tile(crate::world::block::Face::Top));
    let (u0, v0, u1, v1) = atlas.uv_region(tile, 0, 0, 1, 1);
    let dirt = crate::world::block::Block::Dirt;
    let neighbor = atlas.tile_id(dirt, dirt.tile(crate::world::block::Face::Top));
    let triangle = [[-1.0, -1.0, 0.5], [1.0, -1.0, 0.5], [0.0, 1.0, 0.5]].map(|position| Vertex {
        position,
        uv: [
            u0 + (u1 - u0) * (position[0] + 1.0) * 0.5,
            v0 + (v1 - v0) * (position[1] + 1.0) * 0.5,
        ],
        local_uv: [(position[0] + 1.0) * 0.5, (position[1] + 1.0) * 0.5],
        blend_tiles: if blend_edges {
            [neighbor as u32, tile as u32, neighbor as u32, tile as u32]
        } else {
            [tile as u32; 4]
        },
        material: if water_surface { 1.0 } else { 0.0 },
        normal: if water_surface {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        },
        tangent: [1.0, 0.0, 0.0],
        bitangent: [0.0, 1.0, 0.0],
        ao: 1.0,
        light: 1.0,
        texture_region: [0, 0, 1, 1],
    });
    let mut vertices = triangle.to_vec();
    vertices.extend(triangle.map(|mut v| {
        v.position[0] += 4.0;
        v
    }));
    let indices: Vec<u32> = (0..512)
        .flat_map(|_| [0, 1, 2])
        .chain((0..512).flat_map(|_| [3, 4, 5]))
        .collect();
    let batches = mesh_batches(&vertices, &indices);
    let mut visible = Vec::new();
    visible_ranges(&batches, Mat4::IDENTITY, false, &mut visible);
    assert_eq!(visible, vec![0..1536]);
    use wgpu::util::DeviceExt;
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lighting test triangle"),
        contents: bytemuck::cast_slice(&vertices.iter().map(GpuVertex::from).collect::<Vec<_>>()),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("culling test indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let queries = benchmark.then(|| {
        device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("fragment benchmark"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        })
    });
    let resolve = benchmark.then(|| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    });
    let timestamps = benchmark.then(|| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    });
    let mut timings = Vec::new();
    let mut images = Vec::new();
    let cases = if benchmark {
        // One visible triangle, warmup plus repeated frames, then an empty control.
        let mut cases = vec![vec![0..3]; 14];
        cases.push(Vec::new());
        cases
    } else {
        vec![
            std::iter::once(0..indices.len() as u32).collect(),
            visible,
            Vec::new(),
        ]
    };
    for ranges in cases {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test shadow attachment and bindings"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&shadow);
            pass.set_bind_group(0, &bind, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            for range in &ranges {
                pass.draw_indexed(range.clone(), 0, 0..1);
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test shadow sampling in forward pass"),
                timestamp_writes: queries.as_ref().map(|query_set| {
                    wgpu::RenderPassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }
                }),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, &bind, &[]);
            pass.set_bind_group(1, &scene_bind, &[]);
            pass.set_bind_group(2, &shadow_bind, &[]);
            pass.set_pipeline(&sky);
            pass.draw(0..3, 0..1);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            for pipeline in [&voxel, &water] {
                pass.set_pipeline(pipeline);
                for range in ranges.iter().filter(|_| sky_frame.is_none()) {
                    pass.draw_indexed(range.clone(), 0, 0..1);
                }
            }
        }
        if !benchmark && sky_frame.is_none() {
            // Validate bind groups and a depth-less attachment for the
            // additive god-ray pass. The dummy scene is black, so this draw
            // adds no color to the comparison images.
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test godray bindings"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&godray);
            pass.set_bind_group(0, &godray_frame_bind, &[]);
            pass.set_bind_group(1, &scene_bind, &[]);
            pass.draw(0..3, 0..1);
        }
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("culling image readback"),
            size: (row_bytes * dimension) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            target_texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(dimension),
                },
            },
            wgpu::Extent3d {
                width: dimension,
                height: dimension,
                depth_or_array_layers: 1,
            },
        );
        if let (Some(queries), Some(resolve), Some(timestamps)) = (&queries, &resolve, &timestamps)
        {
            encoder.resolve_query_set(queries, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, timestamps, 0, 16);
        }
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU completed lighting passes");
        rx.recv().unwrap().expect("mapped comparison image");
        let mapped = readback
            .slice(..)
            .get_mapped_range()
            .expect("read comparison pixels");
        let pixels: Vec<u8> = mapped
            .chunks_exact(row_bytes as usize)
            .flat_map(|row| row[..dimension as usize * 4].iter().copied())
            .collect();
        images.push(pixels);
        drop(mapped);
        readback.unmap();
        if let Some(timestamps) = &timestamps {
            let (tx, rx) = std::sync::mpsc::channel();
            timestamps
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let data = timestamps.slice(..).get_mapped_range().unwrap();
            let ticks: &[u64] = bytemuck::cast_slice(&data);
            timings.push((ticks[1] - ticks[0]) as f64 * queue.get_timestamp_period() as f64 / 1e6);
            drop(data);
            timestamps.unmap();
        }
    }
    assert_eq!(
        images[0], images[1],
        "culling must preserve rendered pixels"
    );
    if sky_frame.is_none() {
        assert_ne!(
            images[0],
            *images.last().unwrap(),
            "test geometry must contribute visible pixels"
        );
    }
    if benchmark {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/gpu-performance");
        std::fs::create_dir_all(&dir).unwrap();
        let label = if source == VOXEL_SHADER {
            "optimized"
        } else {
            "reference"
        };
        image::save_buffer(
            dir.join(format!("{label}-water-{water_surface}.png")),
            &images[0],
            dimension,
            dimension,
            image::ColorType::Rgba8,
        )
        .unwrap();
        // Exclude the first two frames and the empty control.
        let end = timings.len() - 1;
        let measured = &mut timings[2..end];
        measured.sort_by(f64::total_cmp);
        eprintln!(
            "forward GPU median: {:.4} ms ({} frames)",
            measured[measured.len() / 2],
            measured.len()
        );
    }
    Some(images.remove(0))
}

#[test]
#[ignore = "performance diagnostic: full 192-block view submission counts"]
fn full_view_gpu_submission_counts() {
    use crate::render::voxel::{CHUNK_VIEW_RADIUS, ChunkKey, LodLevel};
    use crate::world::voxel::GeneratedVoxelWorld;
    use glam::{DVec3, Vec3};

    let atlas = atlas::build(Path::new("assets/textures")).expect("atlas");
    let mut world = GeneratedVoxelWorld::new(2026_0904);
    let mut mesh = VoxelMesh::default();
    for cz in -CHUNK_VIEW_RADIUS as i32..=CHUNK_VIEW_RADIUS as i32 {
        for cx in -CHUNK_VIEW_RADIUS as i32..=CHUNK_VIEW_RADIUS as i32 {
            let key = ChunkKey { cx, cz };
            let part = VoxelMesh::build_chunk(
                &mut world,
                key,
                LodLevel::from_distance(key.distance_to(0, 0)),
                DVec3::ZERO,
                &atlas,
                &[],
            );
            let base = mesh.vertices.len() as u32;
            mesh.vertices.extend(part.vertices);
            mesh.indices
                .extend(part.indices.into_iter().map(|i| i + base));
            let base = mesh.transparent_vertices.len() as u32;
            mesh.transparent_vertices.extend(part.transparent_vertices);
            mesh.transparent_indices
                .extend(part.transparent_indices.into_iter().map(|i| i + base));
        }
    }
    let opaque = mesh_batches(&mesh.vertices, &mesh.indices);
    let transparent = mesh_batches(&mesh.transparent_vertices, &mesh.transparent_indices);
    let total_indices = mesh.indices.len() + mesh.transparent_indices.len();
    let vertices = mesh.vertices.len() + mesh.transparent_vertices.len();
    eprintln!(
        "total triangles={}, vertex bytes {} -> {}, upload bytes {} -> {}",
        total_indices / 3,
        vertices * 104,
        vertices * 80,
        vertices * 104 + total_indices * 4,
        vertices * 80 + total_indices * 4
    );
    let camera = Vec3::new(0.0, 85.0, 0.0);
    let projection = Mat4::perspective_rh(70_f32.to_radians(), 16.0 / 9.0, 0.05, 512.0);
    for direction in [Vec3::X, -Vec3::X, Vec3::Z, -Vec3::Z] {
        let matrix =
            projection * Mat4::look_to_rh(camera, (direction - Vec3::Y * 0.2).normalize(), Vec3::Y);
        let mut a = Vec::new();
        let mut b = Vec::new();
        visible_ranges(&opaque, matrix, false, &mut a);
        visible_ranges(&transparent, matrix, false, &mut b);
        let submitted: u32 = a.iter().chain(&b).map(|r| r.end - r.start).sum();
        eprintln!(
            "direction={direction:?}, triangles={}, draws={}, rejected={:.1}%",
            submitted / 3,
            a.len() + b.len(),
            100.0 * (1.0 - submitted as f64 / total_indices as f64)
        );
        assert!(submitted > 0 && (submitted as usize) < total_indices);
    }
    let mut a = Vec::new();
    let mut b = Vec::new();
    let matrix = shadow_matrix(camera, Vec3::Y);
    visible_ranges(&opaque, matrix, true, &mut a);
    visible_ranges(&transparent, matrix, true, &mut b);
    let submitted: u32 = a.iter().chain(&b).map(|r| r.end - r.start).sum();
    eprintln!(
        "zenith shadows: triangles={}, draws={}, rejected={:.1}%",
        submitted / 3,
        a.len() + b.len(),
        100.0 * (1.0 - submitted as f64 / total_indices as f64)
    );
    assert!((submitted as usize) < total_indices);
}

#[test]
fn paired_shadow_taps_preserve_bilinear_kernel() {
    // Exhaust all binary comparison outcomes across a 4x4 footprint,
    // at different subtexel phases, including exact texel boundaries.
    for (sx, sy) in [(0.0, 0.0), (0.13, 0.79), (0.5, 0.5), (0.999, 0.001)] {
        for mask in 0u32..65536 {
            let sample = |x: f64, y: f64| {
                let ix = x.floor() as usize;
                let iy = y.floor() as usize;
                let fx = x.fract();
                let fy = y.fract();
                let value = |x: usize, y: usize| ((mask >> (y * 4 + x)) & 1) as f64;
                (value(ix, iy) * (1.0 - fx) + value(ix + 1, iy) * fx) * (1.0 - fy)
                    + (value(ix, iy + 1) * (1.0 - fx) + value(ix + 1, iy + 1) * fx) * fy
            };
            let mut reference = 0.0;
            for y in 0..3 {
                for x in 0..3 {
                    reference += sample(x as f64 + sx, y as f64 + sy);
                }
            }
            let wx = [2.0 - sx, 1.0 + sx];
            let wy = [2.0 - sy, 1.0 + sy];
            let x = [1.0 / wx[0], 2.0 + sx / wx[1]];
            let y = [1.0 / wy[0], 2.0 + sy / wy[1]];
            let mut paired = 0.0;
            for j in 0..2 {
                for i in 0..2 {
                    paired += sample(x[i], y[j]) * wx[i] * wy[j];
                }
            }
            assert!((reference - paired).abs() < 1e-12);
        }
    }
}

#[test]
fn sky_is_invariant_under_camera_relative_origin_changes() {
    let a = sky_preview_frame(0.55, glam::Vec3::ZERO);
    let b = sky_preview_frame(0.55, glam::Vec3::new(32.0, 16.0, -48.0));
    if let (Some(a), Some(b)) = (
        render_shader_with_sky(VOXEL_SHADER, false, false, false, Some(a)),
        render_shader_with_sky(VOXEL_SHADER, false, false, false, Some(b)),
    ) {
        // Matrix inversion and finite precision can move antialiased cloud edges slightly.
        let mean_error = a
            .iter()
            .zip(&b)
            .map(|(a, b)| a.abs_diff(*b) as f64)
            .sum::<f64>()
            / a.len() as f64;
        assert!(
            mean_error < 0.1,
            "origin shift changed the sky: {mean_error}/255"
        );
    }
}

fn sky_preview_frame(elevation: f32, offset: glam::Vec3) -> FrameUniform {
    let mut frame = FrameUniform::zeroed();
    let sun = glam::Vec3::new(0.35, elevation, -0.8).normalize();
    let day = ((sun.y + 0.16) / 0.32).clamp(0.0, 1.0);
    let twilight = (1.0 - (sun.y.abs() / 0.40).clamp(0.0, 1.0)).powi(2) * day;
    let top = glam::Vec3::new(0.002, 0.004, 0.012).lerp(glam::Vec3::new(0.075, 0.24, 0.52), day);
    let horizon = glam::Vec3::new(0.009, 0.013, 0.028).lerp(glam::Vec3::new(0.52, 0.66, 0.80), day);
    let color =
        glam::Vec3::new(1.0, 0.96, 0.84).lerp(glam::Vec3::new(1.0, 0.48, 0.18), twilight * 0.72);
    let view = Mat4::look_to_rh(offset, glam::Vec3::new(0.0, 0.35, -1.0), glam::Vec3::Y);
    let projection = Mat4::perspective_rh_gl(85_f32.to_radians(), 1.0, 0.05, 384.0);
    frame.view_proj = (projection * view).to_cols_array_2d();
    frame.inv_view_proj = (projection * view).inverse().to_cols_array_2d();
    frame.camera_position = offset.extend(0.0).to_array();
    frame.world_origin = (glam::Vec3::new(0.0, 80.0, 0.0) - offset)
        .extend(0.0)
        .to_array();
    frame.sun_direction = sun.extend(0.0).to_array();
    frame.sun_color = color.extend(day * 0.92).to_array();
    frame.sky_top = top.extend(0.0).to_array();
    frame.sky_horizon = horizon.extend(0.0).to_array();
    frame.fog_color = horizon.extend(0.0).to_array();
    frame.params[0] = day;
    frame.time[0] = 30.0;
    frame.viewport = [512.0, 512.0, 0.0, 0.0];
    frame
}

#[test]
#[ignore = "visual diagnostic: writes day, sunset and night skies under target/sky"]
fn sky_preview_images() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/sky");
    std::fs::create_dir_all(&dir).unwrap();
    for (label, elevation) in [("day", 0.55), ("sunset", 0.04), ("night", -0.55)] {
        let frame = sky_preview_frame(elevation, glam::Vec3::ZERO);
        let pixels = render_shader_with_sky(VOXEL_SHADER, false, false, false, Some(frame))
            .expect("a GPU adapter is required for sky previews");
        image::save_buffer(
            dir.join(format!("{label}.png")),
            &pixels,
            512,
            512,
            image::ColorType::Rgba8,
        )
        .unwrap();
    }
}

#[test]
#[ignore = "GPU timestamp diagnostic: 512x512 cumulus sky"]
fn cumulus_gpu_benchmark() {
    render_shader_with_sky(
        VOXEL_SHADER,
        false,
        false,
        true,
        Some(sky_preview_frame(0.55, glam::Vec3::ZERO)),
    );
}
