//! 应用入口：`winit 0.30 ApplicationHandler`，窗口在 `resumed()` 里建。
//!
//! 启动 UI：MC 风格主菜单，用 egui 绘制。F3 切换调试模式。
//! 当前已经接通：首页 → 世界列表 → 创建世界 → 世界预览。

use std::sync::{Arc, mpsc};
use std::time::Instant;

use egui::epaint as ep;
use glam::{DVec3, Mat4};
use rayon::prelude::*;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::player::inventory::{HOTBAR_SIZE, INVENTORY_SIZE, Inventory};
use crate::player::physics::{Aabb, Player, PlayerInput};
use crate::render::camera::Camera;
use crate::render::gpu::GpuState;
use crate::render::voxel::{VoxelMesh, VoxelMeshWorker};
use crate::world::atlas::Atlas;
use crate::world::block::{Block, Face};
use crate::world::column::{Column, ColumnGen, Y_MAX, Y_MIN};
use crate::world::continent::WorldHeightmap;
use crate::world::voxel::{GeneratedVoxelWorld, VoxelWorld};

const MESH_RADIUS: i64 = 24;
const INITIAL_COLUMNS_RADIUS: i64 = MESH_RADIUS;
const MESH_REBUILD_DISTANCE: i64 = 8;
const BLOCK_REACH: f64 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BlockHit {
    block: (i64, i64, i64),
    place: (i64, i64, i64),
}

#[derive(Clone, Copy)]
struct WorldRenderState {
    in_world: bool,
    underwater: bool,
    view_proj: Mat4,
}

fn install_cjk_font(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let mut cjk_font =
        egui::FontData::from_static(include_bytes!("../assets/fonts/NotoSansCJK-Regular.ttc"));
    // The TTC contains one face per CJK locale; face 2 is Simplified Chinese.
    cjk_font.index = 2;
    fonts
        .font_data
        .insert("NotoSansCJK-Regular".to_owned(), Arc::new(cjk_font));

    // Keep egui's Latin fonts as the primary face and use the bundled CJK
    // face whenever a character is missing from them.
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .get_mut(&family)
            .expect("egui font family")
            .push("NotoSansCJK-Regular".to_owned());
    }
    ctx.set_fonts(fonts);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LauncherScreen {
    Home,
    Worlds,
    CreateWorld,
    CreatingWorld,
    InWorld,
}

enum WorldCreationEvent {
    Progress { amount: f32, stage: &'static str },
    Complete(WorldInfo),
}

#[derive(Clone, Copy)]
struct WorldCreationProgress {
    amount: f32,
    stage: &'static str,
}

#[derive(Clone)]
struct WorldInfo {
    name: String,
    seed: u64,
    spawn_x: i64,
    spawn_z: i64,
    spawn_height: i64,
    initial_columns: Vec<Column>,
}

impl WorldInfo {
    /// 创建世界时生成出生点和一块小预览区，验证 UI 已经接入世界生成器。
    #[cfg(test)]
    fn generate(name: String, seed: u64) -> Self {
        Self::generate_with_progress(name, seed, |_| {})
    }

    fn generate_with_progress(name: String, seed: u64, mut on_progress: impl FnMut(f32)) -> Self {
        let heightmap = WorldHeightmap::new(seed);
        let column_gen = ColumnGen::new(seed);
        on_progress(0.08);

        let (spawn_x, spawn_z, spawn_height) =
            find_safe_spawn_with_progress(&heightmap, &column_gen, |amount| {
                on_progress(0.16 + amount * 0.34)
            });

        let min_x = spawn_x - INITIAL_COLUMNS_RADIUS;
        let max_x = spawn_x + INITIAL_COLUMNS_RADIUS;
        let min_z = spawn_z - INITIAL_COLUMNS_RADIUS;
        let max_z = spawn_z + INITIAL_COLUMNS_RADIUS;
        let coordinates: Vec<_> = (min_z..=max_z)
            .flat_map(|z| (min_x..=max_x).map(move |x| (x, z)))
            .collect();
        // The columns are independent pure-function results, so this is safe
        // to distribute across the Rayon pool. Progress is reported at the
        // phase boundaries rather than from worker threads.
        on_progress(0.50);
        let initial_columns: Vec<Column> = coordinates
            .into_par_iter()
            .map(|(x, z)| column_gen.generate(x, z, heightmap.height(x, z)))
            .collect();
        on_progress(0.96);

        Self {
            name,
            seed,
            spawn_x,
            spawn_z,
            spawn_height,
            initial_columns,
        }
    }
}

/// 找离世界原点最近的安全出生点。
///
/// 按方形边界逐圈搜索，但会继续搜索到当前最佳点的欧氏距离之外，
/// 因此不会因为先遇到方形角落而错过更近的边上位置。水、岩浆等流体
/// 以及任何占据脚部/头部空间的方块都会被排除。
fn find_safe_spawn_with_progress(
    heightmap: &WorldHeightmap,
    column_gen: &ColumnGen,
    mut on_progress: impl FnMut(f32),
) -> (i64, i64, i64) {
    const MAX_SEARCH_RADIUS: i64 = 4096;
    let mut best: Option<(i64, i64, i64, i64)> = None;

    for radius in 0..=MAX_SEARCH_RADIUS {
        if radius == 0 || radius % 64 == 0 {
            on_progress(radius as f32 / MAX_SEARCH_RADIUS as f32);
        }
        let ring: Vec<_> = if radius == 0 {
            vec![(0, 0)]
        } else {
            let mut ring = Vec::with_capacity((radius * 8) as usize);
            for x in -radius..=radius {
                ring.push((x, -radius));
                ring.push((x, radius));
            }
            for z in (-radius + 1)..radius {
                ring.push((-radius, z));
                ring.push((radius, z));
            }
            ring
        };
        if let Some(candidate) = ring
            .into_par_iter()
            .enumerate()
            .filter_map(|(order, (x, z))| {
                spawn_candidate(heightmap, column_gen, x, z)
                    .map(|(distance_squared, x, z, y)| (distance_squared, order, x, z, y))
            })
            .min_by_key(|candidate| (candidate.0, candidate.1))
            && best.is_none_or(|current| candidate.0 < current.0)
        {
            best = Some((candidate.0, candidate.2, candidate.3, candidate.4));
        }

        // Every unvisited point has Chebyshev distance > radius and therefore
        // Euclidean distance > radius. Once that exceeds the best distance,
        // no later ring can contain a nearer point.
        if let Some((distance_squared, ..)) = best
            && radius * radius >= distance_squared
        {
            let (_, x, z, y) = best.expect("安全出生点应当存在");
            return (x, z, y);
        }
    }

    panic!("在半径 {MAX_SEARCH_RADIUS} 内找不到安全出生点；请检查世界生成器");
}

fn spawn_candidate(
    heightmap: &WorldHeightmap,
    column_gen: &ColumnGen,
    x: i64,
    z: i64,
) -> Option<(i64, i64, i64, i64)> {
    let height = heightmap.height(x, z);
    let column = column_gen.generate(x, z, height);
    let spawn_y = safe_spawn_y(&column)?;
    let distance_squared = x * x + z * z;
    Some((distance_squared, x, z, spawn_y))
}

fn safe_spawn_y(column: &Column) -> Option<i64> {
    let floor_y = column.top_solid();
    let floor = column.get(floor_y);
    let feet = column.get(floor_y + 1);
    let head = column.get(floor_y + 2);

    if !floor.is_solid() || feet.is_solid() || feet.is_fluid() || head.is_solid() || head.is_fluid()
    {
        return None;
    }
    Some(floor_y + 1)
}

struct LauncherUi {
    ctx: egui::Context,
    renderer: egui_wgpu::Renderer,
    winit: egui_winit::State,
    window: Arc<Window>,
    time: f64,
    debug: bool,
    screen: LauncherScreen,
    worlds: Vec<WorldInfo>,
    current_world: Option<WorldInfo>,
    creation_receiver: Option<mpsc::Receiver<WorldCreationEvent>>,
    creation_progress: WorldCreationProgress,
    creation_display_progress: f32,
    draft_name: String,
    draft_seed: String,
    form_error: Option<String>,
    exit_requested: bool,
    physics_world: Option<GeneratedVoxelWorld>,
    player: Option<Player>,
    move_forward: bool,
    move_backward: bool,
    move_left: bool,
    move_right: bool,
    space_held: bool,
    swim_down: bool,
    jump_requested: bool,
    physics_accumulator: f64,
    last_frame: Instant,
    camera: Camera,
    voxel_mesh: Option<VoxelMesh>,
    mesh_center: Option<(i64, i64)>,
    mesh_requested_center: Option<(i64, i64)>,
    mesh_needs_rebuild: bool,
    mesh_origin: DVec3,
    mesh_worker: Option<VoxelMeshWorker>,
    inventory: Inventory,
    atlas: Atlas,
    atlas_texture: egui::TextureHandle,
    target_block: Option<BlockHit>,
}

impl LauncherUi {
    fn new(window: Arc<Window>, gpu: &GpuState) -> Self {
        let ctx = egui::Context::default();
        install_cjk_font(&ctx);
        ctx.set_visuals(egui::Visuals {
            override_text_color: Some(egui::Color32::from_rgb(0xE0, 0xE0, 0xE0)),
            ..Default::default()
        });
        let renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            gpu.config.format,
            egui_wgpu::RendererOptions::default(),
        );
        let atlas = gpu.atlas.clone();
        let atlas_image = egui::ColorImage::from_rgba_unmultiplied(
            [atlas.size as usize, atlas.size as usize],
            atlas.image.as_raw(),
        );
        let atlas_texture =
            ctx.load_texture("mc-block-atlas", atlas_image, egui::TextureOptions::NEAREST);
        let viewport_id = ctx.viewport_id();
        let winit = egui_winit::State::new(
            ctx.clone(),
            viewport_id,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        Self {
            ctx,
            renderer,
            winit,
            window,
            time: 0.0,
            debug: false,
            screen: LauncherScreen::Home,
            worlds: Vec::new(),
            current_world: None,
            creation_receiver: None,
            creation_progress: WorldCreationProgress {
                amount: 0.0,
                stage: "准备世界生成器",
            },
            creation_display_progress: 0.0,
            draft_name: "新世界".to_string(),
            draft_seed: "20260904".to_string(),
            form_error: None,
            exit_requested: false,
            physics_world: None,
            player: None,
            move_forward: false,
            move_backward: false,
            move_left: false,
            move_right: false,
            space_held: false,
            swim_down: false,
            jump_requested: false,
            physics_accumulator: 0.0,
            last_frame: Instant::now(),
            camera: Camera::default(),
            voxel_mesh: None,
            mesh_center: None,
            mesh_requested_center: None,
            mesh_needs_rebuild: false,
            mesh_origin: DVec3::ZERO,
            mesh_worker: None,
            inventory: Inventory::new(),
            atlas,
            atlas_texture,
            target_block: None,
        }
    }

    fn toggle_debug(&mut self) {
        self.debug = !self.debug;
    }

    fn input(&mut self, event: &WindowEvent) -> egui_winit::EventResponse {
        self.handle_player_input(event);
        self.winit.on_window_event(&self.window, event)
    }

    fn render(&mut self, gpu: &mut GpuState, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let frame_dt = self.last_frame.elapsed().as_secs_f64().min(0.25);
        self.last_frame = now;
        self.advance_physics(frame_dt);
        self.time += frame_dt;
        self.poll_world_creation();
        self.animate_creation_progress(frame_dt);
        self.update_target_block();
        self.ensure_world_mesh(gpu);
        gpu.upload_highlight(
            self.target_block.map(|target| target.block),
            self.mesh_origin,
        );
        let raw_input = self.winit.take_egui_input(&self.window);
        // 克隆 Context，避免在 run_ui 的借用期间同时借用 LauncherUi 的 ctx 字段。
        let ctx = self.ctx.clone();
        let mut full_output = ctx.run_ui(raw_input, |ui| {
            self.build(ui);
        });
        if self.exit_requested {
            full_output.textures_delta.clear();
            event_loop.exit();
            return;
        }
        self.winit
            .handle_platform_output(&self.window, full_output.platform_output);

        let clipped = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let mut textures_delta = std::mem::take(&mut full_output.textures_delta);
        for (id, deltas) in &textures_delta.set {
            for delta in deltas {
                self.renderer
                    .update_texture(&gpu.device, &gpu.queue, *id, delta);
            }
        }
        let output = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                textures_delta.clear();
                return;
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sd = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gpu.config.width, gpu.config.height],
            pixels_per_point: ctx.pixels_per_point(),
        };
        let in_world = self.screen == LauncherScreen::InWorld;
        let camera_matrix = self.camera_matrix(gpu);
        let underwater = in_world && self.camera_is_underwater();
        let render_state = WorldRenderState {
            in_world,
            underwater,
            view_proj: camera_matrix,
        };
        Self::submit_frame(gpu, &view, &clipped, &sd, &mut self.renderer, render_state);
        gpu.queue.present(output);
        for id in &textures_delta.free {
            self.renderer.free_texture(id);
        }
        textures_delta.clear();
    }

    fn start_world_creation(&mut self, name: String, seed: u64) {
        let (sender, receiver) = mpsc::channel();
        let _ = sender.send(WorldCreationEvent::Progress {
            amount: 0.02,
            stage: "准备世界生成器",
        });

        std::thread::spawn(move || {
            let progress_sender = sender.clone();
            let world = WorldInfo::generate_with_progress(name, seed, |amount| {
                let stage = if amount < 0.16 {
                    "初始化世界"
                } else if amount < 0.50 {
                    "寻找安全出生点"
                } else if amount < 0.92 {
                    "生成地形"
                } else {
                    "整理出生区域"
                };
                let _ = progress_sender.send(WorldCreationEvent::Progress { amount, stage });
            });
            let _ = sender.send(WorldCreationEvent::Complete(world));
        });

        self.creation_receiver = Some(receiver);
        self.creation_progress = WorldCreationProgress {
            amount: 0.02,
            stage: "准备世界生成器",
        };
        self.creation_display_progress = 0.0;
        self.form_error = None;
        self.screen = LauncherScreen::CreatingWorld;
    }

    fn poll_world_creation(&mut self) {
        let Some(receiver) = self.creation_receiver.take() else {
            return;
        };
        let mut completed = None;
        while let Ok(event) = receiver.try_recv() {
            match event {
                WorldCreationEvent::Progress { amount, stage } => {
                    self.creation_progress = WorldCreationProgress { amount, stage };
                }
                WorldCreationEvent::Complete(world) => completed = Some(world),
            }
        }

        if let Some(world) = completed {
            self.worlds.push(world.clone());
            self.creation_progress.amount = 1.0;
            self.creation_display_progress = 1.0;
            self.enter_world(world);
        } else {
            self.creation_receiver = Some(receiver);
        }
    }

    fn animate_creation_progress(&mut self, frame_dt: f64) {
        if self.screen != LauncherScreen::CreatingWorld {
            return;
        }

        // Keep the bar moving smoothly between worker updates without making
        // it outrun the actual generation progress.
        let target = self.creation_progress.amount.clamp(0.0, 1.0);
        let step = (frame_dt as f32 * 3.5).clamp(0.0, 0.08);
        self.creation_display_progress =
            (self.creation_display_progress + step).min(target.max(self.creation_display_progress));
    }

    fn enter_world(&mut self, world: WorldInfo) {
        let spawn = DVec3::new(
            world.spawn_x as f64 + 0.5,
            world.spawn_height as f64,
            world.spawn_z as f64 + 0.5,
        );
        self.physics_world = Some(GeneratedVoxelWorld::with_columns(
            world.seed,
            world.initial_columns.clone(),
        ));
        self.player = Some(Player::new(spawn));
        self.physics_accumulator = 0.0;
        self.space_held = false;
        self.swim_down = false;
        self.jump_requested = false;
        self.inventory.close();
        self.camera.reset();
        self.voxel_mesh = None;
        self.mesh_center = None;
        self.mesh_requested_center = None;
        self.mesh_needs_rebuild = true;
        self.mesh_origin = spawn;
        self.mesh_worker = None;
        self.target_block = None;
        let _ = self
            .window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined));
        self.window.set_cursor_visible(false);
        self.current_world = Some(world);
        self.screen = LauncherScreen::InWorld;
    }

    fn leave_world(&mut self) {
        self.screen = LauncherScreen::Worlds;
        self.move_forward = false;
        self.move_backward = false;
        self.move_left = false;
        self.move_right = false;
        self.space_held = false;
        self.swim_down = false;
        self.jump_requested = false;
        self.inventory.close();
        self.mesh_worker = None;
        self.mesh_requested_center = None;
        self.mesh_needs_rebuild = false;
        self.target_block = None;
        let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        self.window.set_cursor_visible(true);
    }

    fn handle_player_input(&mut self, event: &WindowEvent) {
        if self.screen != LauncherScreen::InWorld {
            return;
        }
        if let WindowEvent::MouseInput {
            state: ElementState::Pressed,
            button,
            ..
        } = event
        {
            if !self.inventory.open {
                match button {
                    MouseButton::Left => self.mine_targeted_block(),
                    MouseButton::Right => self.place_targeted_block(),
                    _ => {}
                }
            }
            return;
        }
        let WindowEvent::KeyboardInput { event, .. } = event else {
            return;
        };
        let PhysicalKey::Code(code) = event.physical_key else {
            return;
        };
        let pressed = event.state == ElementState::Pressed;
        if code == KeyCode::Escape && pressed && !event.repeat {
            if self.inventory.open {
                self.close_inventory();
            } else {
                self.leave_world();
            }
            return;
        }
        if code == KeyCode::KeyE && pressed && !event.repeat {
            self.toggle_inventory();
            return;
        }
        if self.inventory.open {
            return;
        }
        if pressed && !event.repeat {
            match code {
                KeyCode::Digit1
                | KeyCode::Digit2
                | KeyCode::Digit3
                | KeyCode::Digit4
                | KeyCode::Digit5
                | KeyCode::Digit6
                | KeyCode::Digit7
                | KeyCode::Digit8
                | KeyCode::Digit9 => {
                    let slot = match code {
                        KeyCode::Digit1 => 0,
                        KeyCode::Digit2 => 1,
                        KeyCode::Digit3 => 2,
                        KeyCode::Digit4 => 3,
                        KeyCode::Digit5 => 4,
                        KeyCode::Digit6 => 5,
                        KeyCode::Digit7 => 6,
                        KeyCode::Digit8 => 7,
                        KeyCode::Digit9 => 8,
                        _ => unreachable!(),
                    };
                    self.inventory.select(INVENTORY_SIZE - HOTBAR_SIZE + slot);
                }
                _ => {}
            }
        }
        match code {
            KeyCode::KeyW => self.move_forward = pressed,
            KeyCode::KeyS => self.move_backward = pressed,
            KeyCode::KeyA => self.move_left = pressed,
            KeyCode::KeyD => self.move_right = pressed,
            KeyCode::Space => {
                self.space_held = pressed;
                if pressed && !event.repeat {
                    self.jump_requested = true;
                }
            }
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.swim_down = pressed,
            _ => {}
        }
    }

    fn mouse_motion(&mut self, delta: (f64, f64)) {
        if self.screen == LauncherScreen::InWorld && !self.inventory.open {
            self.camera.mouse_motion(delta.0, delta.1);
        }
    }

    fn update_target_block(&mut self) {
        if self.inventory.open {
            self.target_block = None;
            return;
        }
        let (Some(player), Some(world)) = (self.player.as_ref(), self.physics_world.as_mut())
        else {
            self.target_block = None;
            return;
        };
        self.target_block = raycast_block(
            world,
            self.camera.eye_position(player),
            self.camera.forward(),
            BLOCK_REACH,
        );
    }

    fn mine_targeted_block(&mut self) {
        self.update_target_block();
        let Some(target) = self.target_block else {
            return;
        };
        let Some(world) = self.physics_world.as_mut() else {
            return;
        };
        let block = world.block_at(target.block.0, target.block.1, target.block.2);
        if block.def().hardness.is_none() {
            return;
        }
        world.set_block(target.block.0, target.block.1, target.block.2, Block::Air);
        self.inventory.give(block);
        self.mesh_needs_rebuild = true;
        self.mesh_requested_center = None;
    }

    fn place_targeted_block(&mut self) {
        self.update_target_block();
        let Some(target) = self.target_block else {
            return;
        };
        let Some(block) = self.inventory.selected_stack().map(|stack| stack.block) else {
            return;
        };
        let Some(player) = self.player.as_ref() else {
            return;
        };
        let place_aabb = Aabb::new(
            DVec3::new(
                target.place.0 as f64,
                target.place.1 as f64,
                target.place.2 as f64,
            ),
            DVec3::new(
                target.place.0 as f64 + 1.0,
                target.place.1 as f64 + 1.0,
                target.place.2 as f64 + 1.0,
            ),
        );
        if !(Y_MIN..=Y_MAX).contains(&target.place.1) {
            return;
        }
        let place_is_empty_or_fluid = self.physics_world.as_mut().is_some_and(|world| {
            let target_block = world.block_at(target.place.0, target.place.1, target.place.2);
            target_block == Block::Air || target_block.is_fluid()
        });
        if overlaps_aabb(player.aabb(), place_aabb) || !place_is_empty_or_fluid {
            return;
        }
        if let Some(world) = self.physics_world.as_mut() {
            world.set_block(target.place.0, target.place.1, target.place.2, block);
        }
        self.inventory.take_selected();
        self.mesh_needs_rebuild = true;
        self.mesh_requested_center = None;
    }

    fn toggle_inventory(&mut self) {
        if self.inventory.open {
            self.close_inventory();
            return;
        }
        self.inventory.toggle();
        self.stop_player_input();
        let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        self.window.set_cursor_visible(true);
    }

    fn close_inventory(&mut self) {
        self.inventory.close();
        let _ = self
            .window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined));
        self.window.set_cursor_visible(false);
    }

    fn stop_player_input(&mut self) {
        self.move_forward = false;
        self.move_backward = false;
        self.move_left = false;
        self.move_right = false;
        self.space_held = false;
        self.swim_down = false;
        self.jump_requested = false;
    }

    fn player_input(&self, jump: bool) -> PlayerInput {
        let forward = self.camera.forward();
        let right = DVec3::new(self.camera.yaw.cos(), 0.0, self.camera.yaw.sin());
        let direction = right * (self.move_right as i8 - self.move_left as i8) as f64
            - forward * (self.move_backward as i8 - self.move_forward as i8) as f64;
        PlayerInput {
            move_x: direction.x,
            move_z: direction.z,
            jump,
            swim_up: self.space_held,
            swim_down: self.swim_down,
        }
    }

    /// 以 60 TPS 固定步长推进物理，渲染帧率变化不会改变移动和重力结果。
    fn advance_physics(&mut self, frame_dt: f64) {
        if self.screen != LauncherScreen::InWorld || self.inventory.open {
            return;
        }

        const FIXED_DT: f64 = 1.0 / 60.0;
        self.physics_accumulator = (self.physics_accumulator + frame_dt).min(0.25);
        let mut ticks = 0;
        while self.physics_accumulator >= FIXED_DT {
            let input = self.player_input(self.jump_requested && ticks == 0);
            let (Some(world), Some(player)) = (self.physics_world.as_mut(), self.player.as_mut())
            else {
                break;
            };
            player.step(world, input, FIXED_DT);
            self.physics_accumulator -= FIXED_DT;
            ticks += 1;
        }
        if ticks > 0 {
            self.jump_requested = false;
        }
    }

    fn ensure_world_mesh(&mut self, gpu: &mut GpuState) {
        if self.screen != LauncherScreen::InWorld {
            return;
        }
        let Some(seed) = self.current_world.as_ref().map(|world| world.seed) else {
            return;
        };

        if self.mesh_worker.is_none() {
            let initial_columns = self
                .current_world
                .as_ref()
                .map(|world| world.initial_columns.clone())
                .unwrap_or_default();
            self.mesh_worker = Some(VoxelMeshWorker::new(
                seed,
                Arc::new(gpu.atlas.clone()),
                initial_columns,
            ));
        }

        if let Some(worker) = self.mesh_worker.as_mut()
            && let Some(result) = worker.try_take_latest()
        {
            gpu.upload_mesh(&result.mesh);
            self.mesh_origin = result.origin;
            self.mesh_center = Some(result.center);
            self.mesh_requested_center = None;
            self.voxel_mesh = Some(result.mesh);
        }

        let Some(player) = self.player.as_ref() else {
            return;
        };
        let center = (
            player.position.x.floor() as i64,
            player.position.z.floor() as i64,
        );
        let far_enough = |a: (i64, i64), b: (i64, i64)| {
            (a.0 - b.0).abs() >= MESH_REBUILD_DISTANCE || (a.1 - b.1).abs() >= MESH_REBUILD_DISTANCE
        };
        let needs_request = if self.mesh_needs_rebuild {
            self.mesh_requested_center.is_none()
        } else {
            match (self.mesh_center, self.mesh_requested_center) {
                (None, None) => true,
                (None, Some(requested)) => far_enough(requested, center),
                (Some(previous), _) if !far_enough(previous, center) => false,
                (Some(_), Some(requested)) => far_enough(requested, center),
                (Some(_), None) => true,
            }
        };
        if needs_request && let Some(worker) = self.mesh_worker.as_mut() {
            let edits = self
                .physics_world
                .as_ref()
                .map(|world| {
                    world
                        .edits()
                        .map(|(position, &block)| (*position, block))
                        .collect()
                })
                .unwrap_or_default();
            worker.request(center, MESH_RADIUS, player.position, edits);
            self.mesh_requested_center = Some(center);
            self.mesh_needs_rebuild = false;
        }
    }

    fn camera_matrix(&self, gpu: &GpuState) -> Mat4 {
        let (Some(player), Some(_mesh)) = (self.player.as_ref(), self.voxel_mesh.as_ref()) else {
            return Mat4::IDENTITY;
        };
        self.camera.view_projection(
            player,
            self.mesh_origin,
            gpu.config.width,
            gpu.config.height,
        )
    }

    fn camera_is_underwater(&mut self) -> bool {
        let Some(player) = self.player.as_ref() else {
            return false;
        };
        let eye = self.camera.eye_position(player);
        let Some(world) = self.physics_world.as_mut() else {
            return false;
        };
        world
            .block_at(
                eye.x.floor() as i64,
                eye.y.floor() as i64,
                eye.z.floor() as i64,
            )
            .is_fluid()
    }

    fn submit_frame(
        gpu: &mut GpuState,
        view: &wgpu::TextureView,
        clipped: &[ep::ClippedPrimitive],
        sd: &egui_wgpu::ScreenDescriptor,
        renderer: &mut egui_wgpu::Renderer,
        render_state: WorldRenderState,
    ) {
        let WorldRenderState {
            in_world,
            underwater,
            view_proj,
        } = render_state;
        if in_world {
            gpu.update_camera(view_proj, underwater);
        }
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mc frame"),
            });
        // pass 1: clear and, in-world, render the camera-relative voxel mesh.
        {
            let p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(if in_world { "mc world" } else { "mc clear" }),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if underwater {
                            wgpu::Color {
                                r: 0.12,
                                g: 0.48,
                                b: 0.56,
                                a: 1.0,
                            }
                        } else if in_world {
                            crate::render::gpu::WORLD_CLEAR
                        } else {
                            wgpu::Color {
                                r: 0.05,
                                g: 0.07,
                                b: 0.12,
                                a: 1.0,
                            }
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: in_world.then_some(
                    wgpu::RenderPassDepthStencilAttachment {
                        view: gpu.depth_view(),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    },
                ),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if in_world {
                let mut p = p.forget_lifetime();
                gpu.draw_voxels(&mut p);
                gpu.draw_transparent_voxels(&mut p);
                gpu.draw_highlight(&mut p);
            }
        }
        // pass 2: egui overlay
        renderer.update_buffers(&gpu.device, &gpu.queue, &mut enc, clipped, sd);
        {
            let p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mc launcher overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            renderer.render(&mut p.forget_lifetime(), clipped, sd);
        }
        gpu.queue.submit(std::iter::once(enc.finish()));
    }

    fn build(&mut self, ui: &mut egui::Ui) {
        let screen = ui.ctx().viewport_rect();
        if self.screen != LauncherScreen::InWorld {
            Self::draw_backdrop(ui, screen);
        }

        match self.screen {
            LauncherScreen::Home => self.draw_home(ui, screen),
            LauncherScreen::Worlds => self.draw_worlds(ui, screen),
            LauncherScreen::CreateWorld => self.draw_create_world(ui, screen),
            LauncherScreen::CreatingWorld => self.draw_creating_world(ui, screen),
            LauncherScreen::InWorld => self.draw_in_world(ui, screen),
        }

        if self.debug {
            Self::draw_debug(ui);
        }
    }

    fn draw_backdrop(ui: &egui::Ui, screen: ep::Rect) {
        let painter = ui.painter();
        painter.rect_filled(screen, 0.0, egui::Color32::from_rgb(0x5C, 0x3D, 0x1E));
        painter.rect_filled(
            screen,
            0.0,
            egui::Color32::from_rgba_premultiplied(0, 0, 0, 60),
        );

        let grid = egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(0, 0, 0, 40));
        let w = screen.width() as usize;
        let h = screen.height() as usize;
        for gx in (0..w).step_by(64) {
            let x = gx as f32;
            painter.line_segment([ep::Pos2::new(x, 0.0), ep::Pos2::new(x, h as f32)], grid);
        }
        for gy in (0..h).step_by(64) {
            let y = gy as f32;
            painter.line_segment([ep::Pos2::new(0.0, y), ep::Pos2::new(w as f32, y)], grid);
        }
    }

    fn panel(ui: &egui::Ui, rect: ep::Rect) {
        let painter = ui.painter();
        painter.rect_filled(
            rect,
            2.0,
            egui::Color32::from_rgba_premultiplied(0, 0, 0, 175),
        );
        painter.rect_stroke(
            rect,
            2.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0x72, 0x52, 0x2F)),
            egui::StrokeKind::Inside,
        );
    }

    fn text(
        ui: &egui::Ui,
        pos: ep::Pos2,
        text: impl Into<String>,
        size: f32,
        color: egui::Color32,
    ) {
        ui.painter().text(
            pos,
            egui::Align2::CENTER_CENTER,
            text.into(),
            egui::FontId::new(size, egui::FontFamily::Proportional),
            color,
        );
    }

    fn left_text(
        ui: &egui::Ui,
        pos: ep::Pos2,
        text: impl Into<String>,
        size: f32,
        color: egui::Color32,
    ) {
        ui.painter().text(
            pos,
            egui::Align2::LEFT_CENTER,
            text.into(),
            egui::FontId::new(size, egui::FontFamily::Proportional),
            color,
        );
    }

    fn menu_button(ui: &mut egui::Ui, rect: ep::Rect, label: &str, enabled: bool) -> bool {
        let response = ui.allocate_rect(rect, egui::Sense::click());
        let hover = enabled && response.hovered();
        let (top, bottom, text_color) = if !enabled {
            (
                egui::Color32::from_rgb(0x55, 0x55, 0x55),
                egui::Color32::from_rgb(0x33, 0x33, 0x33),
                egui::Color32::from_rgb(0x80, 0x80, 0x80),
            )
        } else if hover {
            (
                egui::Color32::from_rgb(0x70, 0x70, 0xA0),
                egui::Color32::from_rgb(0x40, 0x40, 0x60),
                egui::Color32::from_rgb(0xFF, 0xFF, 0xA0),
            )
        } else {
            (
                egui::Color32::from_rgb(0x6B, 0x6B, 0x6B),
                egui::Color32::from_rgb(0x40, 0x40, 0x40),
                egui::Color32::from_rgb(0xE0, 0xE0, 0xE0),
            )
        };

        let painter = ui.painter();
        painter.rect_filled(rect, 1.0, top);
        painter.rect_filled(
            ep::Rect::from_min_max(ep::Pos2::new(rect.min.x, rect.center().y), rect.max),
            1.0,
            bottom,
        );
        painter.rect_stroke(
            rect,
            1.0,
            egui::Stroke::new(
                1.0,
                if hover {
                    egui::Color32::from_rgb(0xAA, 0xAA, 0xD0)
                } else {
                    egui::Color32::from_rgb(0x4A, 0x4A, 0x4A)
                },
            ),
            egui::StrokeKind::Inside,
        );
        Self::text(ui, rect.center(), label, 17.0, text_color);
        response.clicked() && enabled
    }

    fn draw_home(&mut self, ui: &mut egui::Ui, screen: ep::Rect) {
        let w = screen.width();
        let h = screen.height();
        let cx = screen.center().x;
        let pw = w.clamp(240.0, 380.0);
        let ph = h * 0.55;
        let px = cx - pw / 2.0;
        let py = h * 0.18;
        Self::panel(
            ui,
            ep::Rect::from_min_size(ep::Pos2::new(px, py), ep::Vec2::new(pw, ph)),
        );

        let float = (self.time * 0.8).sin() as f32 * 4.0;
        let logo_y = py + 30.0 + float;
        Self::text(
            ui,
            ep::Pos2::new(cx + 3.0, logo_y + 3.0),
            "mc",
            80.0,
            egui::Color32::from_rgb(0x4A, 0x2A, 0x0A),
        );
        Self::text(
            ui,
            ep::Pos2::new(cx, logo_y),
            "mc",
            80.0,
            egui::Color32::from_rgb(0xD4, 0xA0, 0x3C),
        );
        Self::text(
            ui,
            ep::Pos2::new(cx, logo_y + 61.0),
            "mc 0.1.0 · world launcher",
            13.0,
            egui::Color32::GRAY,
        );

        let bw = pw * 0.62;
        let bh = 34.0;
        let gap = 5.0;
        let bx = cx - bw / 2.0;
        let by0 = logo_y + 124.0;
        let labels = ["单人游戏", "多人游戏", "选项…", "退出"];
        let enabled = [true, false, false, true];
        for (i, label) in labels.iter().enumerate() {
            let rect = ep::Rect::from_min_size(
                ep::Pos2::new(bx, by0 + i as f32 * (bh + gap)),
                ep::Vec2::new(bw, bh),
            );
            if Self::menu_button(ui, rect, label, enabled[i]) {
                match i {
                    0 => {
                        self.form_error = None;
                        self.screen = LauncherScreen::Worlds;
                    }
                    3 => self.exit_requested = true,
                    _ => {}
                }
            }
        }

        let bar_h = 26.0;
        let bar = ep::Rect::from_min_size(ep::Pos2::new(0.0, h - bar_h), ep::Vec2::new(w, bar_h));
        ui.painter().rect_filled(
            bar,
            0.0,
            egui::Color32::from_rgba_premultiplied(0, 0, 0, 160),
        );
        Self::left_text(
            ui,
            ep::Pos2::new(8.0, h - bar_h / 2.0),
            "© mc project · rust + wgpu",
            11.0,
            egui::Color32::from_rgba_premultiplied(255, 255, 255, 120),
        );
        Self::left_text(
            ui,
            ep::Pos2::new(w - 92.0, h - bar_h / 2.0),
            "seed 20260904",
            11.0,
            egui::Color32::from_rgba_premultiplied(255, 255, 255, 120),
        );
    }

    fn draw_worlds(&mut self, ui: &mut egui::Ui, screen: ep::Rect) {
        let w = screen.width();
        let h = screen.height();
        let cx = screen.center().x;
        let pw = w.clamp(320.0, 620.0);
        let ph = (h * 0.72).clamp(390.0, 600.0);
        let px = cx - pw / 2.0;
        let py = (h - ph) / 2.0;
        let panel = ep::Rect::from_min_size(ep::Pos2::new(px, py), ep::Vec2::new(pw, ph));
        Self::panel(ui, panel);
        Self::text(
            ui,
            ep::Pos2::new(cx, py + 34.0),
            "选择世界",
            28.0,
            egui::Color32::WHITE,
        );
        Self::text(
            ui,
            ep::Pos2::new(cx, py + 64.0),
            "选择一个已有世界，或创建新的世界",
            13.0,
            egui::Color32::from_rgb(0xB0, 0xB0, 0xB0),
        );

        let mut enter_index = None;
        if self.worlds.is_empty() {
            Self::text(
                ui,
                ep::Pos2::new(cx, py + 145.0),
                "还没有世界",
                18.0,
                egui::Color32::from_rgb(0xC0, 0xC0, 0xC0),
            );
            Self::text(
                ui,
                ep::Pos2::new(cx, py + 174.0),
                "创建一个世界开始游戏",
                13.0,
                egui::Color32::from_rgb(0x8A, 0x8A, 0x8A),
            );
        } else {
            let row_x = px + 24.0;
            let row_w = pw - 48.0;
            for (i, world) in self.worlds.iter().enumerate() {
                let row_y = py + 92.0 + i as f32 * 70.0;
                let row = ep::Rect::from_min_size(
                    ep::Pos2::new(row_x, row_y),
                    ep::Vec2::new(row_w, 58.0),
                );
                ui.painter().rect_filled(
                    row,
                    2.0,
                    egui::Color32::from_rgba_premultiplied(35, 25, 15, 220),
                );
                Self::left_text(
                    ui,
                    ep::Pos2::new(row.min.x + 14.0, row.min.y + 18.0),
                    world.name.clone(),
                    17.0,
                    egui::Color32::WHITE,
                );
                Self::left_text(
                    ui,
                    ep::Pos2::new(row.min.x + 14.0, row.min.y + 41.0),
                    format!(
                        "种子 {} · 出生点 {}, {}, {}",
                        world.seed, world.spawn_x, world.spawn_height, world.spawn_z
                    ),
                    11.0,
                    egui::Color32::from_rgb(0xA0, 0xA0, 0xA0),
                );
                let enter = ep::Rect::from_min_size(
                    ep::Pos2::new(row.max.x - 92.0, row.min.y + 12.0),
                    ep::Vec2::new(78.0, 34.0),
                );
                if Self::menu_button(ui, enter, "进入", true) {
                    enter_index = Some(i);
                }
            }
        }

        if let Some(index) = enter_index
            && let Some(world) = self.worlds.get(index).cloned()
        {
            self.enter_world(world);
        }

        let button_w = (pw - 64.0) / 2.0;
        let bottom_y = py + ph - 58.0;
        let create = ep::Rect::from_min_size(
            ep::Pos2::new(px + 24.0, bottom_y),
            ep::Vec2::new(button_w, 34.0),
        );
        let back = ep::Rect::from_min_size(
            ep::Pos2::new(px + 40.0 + button_w, bottom_y),
            ep::Vec2::new(button_w, 34.0),
        );
        if Self::menu_button(ui, create, "创建新世界", true) {
            self.form_error = None;
            self.screen = LauncherScreen::CreateWorld;
        }
        if Self::menu_button(ui, back, "返回", true) {
            self.screen = LauncherScreen::Home;
        }
    }

    fn draw_create_world(&mut self, ui: &mut egui::Ui, screen: ep::Rect) {
        let w = screen.width();
        let h = screen.height();
        let cx = screen.center().x;
        let pw = w.clamp(320.0, 520.0);
        let ph = (h - 40.0).clamp(390.0, 500.0);
        let px = cx - pw / 2.0;
        let py = (h - ph) / 2.0;
        Self::panel(
            ui,
            ep::Rect::from_min_size(ep::Pos2::new(px, py), ep::Vec2::new(pw, ph)),
        );
        Self::text(
            ui,
            ep::Pos2::new(cx, py + 36.0),
            "创建新世界",
            28.0,
            egui::Color32::WHITE,
        );

        let label_x = px + 32.0;
        let field_x = px + 32.0;
        let field_w = pw - 64.0;
        Self::left_text(
            ui,
            ep::Pos2::new(label_x, py + 86.0),
            "世界名称",
            14.0,
            egui::Color32::from_rgb(0xD0, 0xD0, 0xD0),
        );
        let name_rect = ep::Rect::from_min_size(
            ep::Pos2::new(field_x, py + 101.0),
            ep::Vec2::new(field_w, 36.0),
        );
        ui.put(
            name_rect,
            egui::TextEdit::singleline(&mut self.draft_name).hint_text("世界名称"),
        );

        Self::left_text(
            ui,
            ep::Pos2::new(label_x, py + 162.0),
            "世界种子",
            14.0,
            egui::Color32::from_rgb(0xD0, 0xD0, 0xD0),
        );
        let seed_rect = ep::Rect::from_min_size(
            ep::Pos2::new(field_x, py + 177.0),
            ep::Vec2::new(field_w, 36.0),
        );
        ui.put(
            seed_rect,
            egui::TextEdit::singleline(&mut self.draft_seed).hint_text("请输入数字种子"),
        );
        Self::left_text(
            ui,
            ep::Pos2::new(label_x, py + 237.0),
            "种子会决定地形生成结果；相同种子会得到相同世界。",
            12.0,
            egui::Color32::from_rgb(0x9A, 0x9A, 0x9A),
        );

        if let Some(error) = self.form_error.as_deref() {
            Self::left_text(
                ui,
                ep::Pos2::new(label_x, py + 275.0),
                error,
                13.0,
                egui::Color32::from_rgb(0xFF, 0xA0, 0x80),
            );
        }

        let button_w = (pw - 64.0) / 2.0;
        let bottom_y = py + ph - 54.0;
        let create = ep::Rect::from_min_size(
            ep::Pos2::new(px + 24.0, bottom_y),
            ep::Vec2::new(button_w, 34.0),
        );
        let back = ep::Rect::from_min_size(
            ep::Pos2::new(px + 40.0 + button_w, bottom_y),
            ep::Vec2::new(button_w, 34.0),
        );
        if Self::menu_button(ui, create, "创建并进入", true) {
            let name = self.draft_name.trim().to_string();
            let seed = self.draft_seed.trim().parse::<u64>();
            if name.is_empty() {
                self.form_error = Some("世界名称不能为空".to_string());
            } else if seed.is_err() {
                self.form_error = Some("世界种子必须是非负整数".to_string());
            } else {
                self.start_world_creation(name, seed.expect("seed checked above"));
            }
        }
        if Self::menu_button(ui, back, "返回", true) {
            self.form_error = None;
            self.screen = LauncherScreen::Worlds;
        }
    }

    fn draw_creating_world(&self, ui: &mut egui::Ui, screen: ep::Rect) {
        let painter = ui.painter();
        painter.rect_filled(screen, 0.0, egui::Color32::from_rgb(0x18, 0x12, 0x0C));

        // A restrained animated glow keeps the screen visibly alive while the
        // worker is busy, matching the chunky, utilitarian MC loading feel.
        let pulse = ((self.time * 2.2).sin() * 0.5 + 0.5) as f32;
        let center = screen.center();
        Self::text(
            ui,
            ep::Pos2::new(center.x, center.y - 78.0),
            "正在生成世界",
            28.0,
            egui::Color32::WHITE,
        );
        Self::text(
            ui,
            ep::Pos2::new(center.x, center.y - 39.0),
            self.creation_progress.stage,
            14.0,
            egui::Color32::from_rgb(0xB8, 0xB8, 0xB8),
        );

        let bar = ep::Rect::from_center_size(
            ep::Pos2::new(center.x, center.y + 4.0),
            egui::Vec2::new(360.0, 22.0),
        );
        painter.rect_filled(bar, 1.0, egui::Color32::from_rgb(0x08, 0x08, 0x08));
        painter.rect_stroke(
            bar,
            1.0,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(0x6A, 0x6A, 0x6A)),
            egui::StrokeKind::Inside,
        );

        let fill_width = (bar.width() - 6.0) * self.creation_display_progress;
        if fill_width > 0.0 {
            let fill = ep::Rect::from_min_size(
                bar.min + egui::Vec2::splat(3.0),
                egui::Vec2::new(fill_width, bar.height() - 6.0),
            );
            painter.rect_filled(fill, 0.0, egui::Color32::from_rgb(0x62, 0xA5, 0x3D));

            let shimmer_x = fill.left() + fill.width() * pulse;
            let shimmer = ep::Rect::from_min_max(
                ep::Pos2::new((shimmer_x - 14.0).max(fill.left()), fill.top()),
                ep::Pos2::new((shimmer_x + 14.0).min(fill.right()), fill.bottom()),
            );
            painter.rect_filled(
                shimmer,
                0.0,
                egui::Color32::from_rgba_premultiplied(220, 255, 180, 42),
            );
        }

        let percent = self.creation_display_progress * 100.0;
        Self::text(
            ui,
            ep::Pos2::new(center.x, center.y + 50.0),
            format!("{percent:02.0}%"),
            13.0,
            egui::Color32::from_rgb(0xA0, 0xA0, 0xA0),
        );

        let dot_count = (self.time * 2.0).floor() as usize % 4;
        Self::text(
            ui,
            ep::Pos2::new(center.x, center.y + 84.0),
            format!("请稍候{}", ".".repeat(dot_count)),
            13.0,
            egui::Color32::from_rgba_premultiplied(255, 255, 255, (130.0 + pulse * 80.0) as u8),
        );
    }

    fn draw_in_world(&mut self, ui: &mut egui::Ui, screen: ep::Rect) {
        let painter = ui.painter();
        let center = screen.center();
        let crosshair = egui::Color32::from_rgba_premultiplied(255, 255, 255, 210);
        painter.line_segment(
            [
                ep::Pos2::new(center.x - 8.0, center.y),
                ep::Pos2::new(center.x + 8.0, center.y),
            ],
            egui::Stroke::new(1.5, crosshair),
        );
        painter.line_segment(
            [
                ep::Pos2::new(center.x, center.y - 8.0),
                ep::Pos2::new(center.x, center.y + 8.0),
            ],
            egui::Stroke::new(1.5, crosshair),
        );
        Self::left_text(
            ui,
            ep::Pos2::new(14.0, 18.0),
            "WASD 移动 · 鼠标视角 · 左键挖掘 · 右键放置 · Space 跳跃/上浮 · Shift 下潜 · E 物品栏 · Esc 返回",
            13.0,
            egui::Color32::from_rgba_premultiplied(255, 255, 255, 220),
        );
        if let Some(player) = self.player.as_ref() {
            Self::left_text(
                ui,
                ep::Pos2::new(14.0, 42.0),
                format!(
                    "位置 {:.1}, {:.1}, {:.1} · {} · 网格 {} 面",
                    player.position.x,
                    player.position.y,
                    player.position.z,
                    if player.on_ground { "地面" } else { "空中" },
                    self.voxel_mesh
                        .as_ref()
                        .map(|mesh| mesh.indices.len() / 3)
                        .unwrap_or(0)
                ),
                12.0,
                egui::Color32::from_rgba_premultiplied(255, 255, 255, 180),
            );
        }
        if self.inventory.open {
            self.draw_inventory(ui, screen);
        } else {
            self.draw_hotbar(ui, screen);
        }
    }

    fn draw_hotbar(&mut self, ui: &mut egui::Ui, screen: ep::Rect) {
        const SLOT: f32 = 44.0;
        const GAP: f32 = 4.0;
        let total_width = HOTBAR_SIZE as f32 * SLOT + (HOTBAR_SIZE - 1) as f32 * GAP;
        let x = screen.center().x - total_width / 2.0;
        let y = screen.max.y - SLOT - 22.0;
        for index in 0..HOTBAR_SIZE {
            let rect = ep::Rect::from_min_size(
                ep::Pos2::new(x + index as f32 * (SLOT + GAP), y),
                ep::Vec2::splat(SLOT),
            );
            self.draw_inventory_slot(ui, INVENTORY_SIZE - HOTBAR_SIZE + index, rect);
        }
        Self::text(
            ui,
            ep::Pos2::new(screen.center().x, y - 12.0),
            "1–9 选择 · E 打开物品栏",
            11.0,
            egui::Color32::from_rgba_premultiplied(255, 255, 255, 165),
        );
    }

    fn draw_inventory(&mut self, ui: &mut egui::Ui, screen: ep::Rect) {
        const SLOT: f32 = 42.0;
        const GAP: f32 = 4.0;
        const COLUMNS: usize = HOTBAR_SIZE;
        let grid_width = COLUMNS as f32 * SLOT + (COLUMNS - 1) as f32 * GAP;
        let panel_width = grid_width + 42.0;
        let panel_height: f32 = 274.0;
        let panel = ep::Rect::from_center_size(
            screen.center(),
            ep::Vec2::new(panel_width, panel_height.min(screen.height() - 24.0)),
        );
        ui.painter().rect_filled(
            screen,
            0.0,
            egui::Color32::from_rgba_premultiplied(0, 0, 0, 135),
        );
        Self::panel(ui, panel);
        Self::left_text(
            ui,
            ep::Pos2::new(panel.min.x + 20.0, panel.min.y + 24.0),
            "物品栏",
            20.0,
            egui::Color32::WHITE,
        );
        Self::left_text(
            ui,
            ep::Pos2::new(panel.max.x - 20.0, panel.min.y + 24.0),
            "E / Esc 关闭",
            11.0,
            egui::Color32::from_rgb(0xB0, 0xB0, 0xB0),
        );

        let x = panel.min.x + 21.0;
        let main_y = panel.min.y + 48.0;
        for row in 0..3 {
            for column in 0..COLUMNS {
                let index = row * COLUMNS + column;
                let rect = ep::Rect::from_min_size(
                    ep::Pos2::new(
                        x + column as f32 * (SLOT + GAP),
                        main_y + row as f32 * (SLOT + GAP),
                    ),
                    egui::Vec2::splat(SLOT),
                );
                self.draw_inventory_slot(ui, index, rect);
            }
        }

        let hotbar_y = panel.max.y - SLOT - 20.0;
        for column in 0..HOTBAR_SIZE {
            let rect = ep::Rect::from_min_size(
                ep::Pos2::new(x + column as f32 * (SLOT + GAP), hotbar_y),
                egui::Vec2::splat(SLOT),
            );
            self.draw_inventory_slot(ui, INVENTORY_SIZE - HOTBAR_SIZE + column, rect);
        }
        Self::left_text(
            ui,
            ep::Pos2::new(panel.min.x + 20.0, panel.max.y - 5.0),
            "点击格子选择方块",
            11.0,
            egui::Color32::from_rgb(0xB0, 0xB0, 0xB0),
        );
    }

    fn draw_inventory_slot(&mut self, ui: &mut egui::Ui, index: usize, rect: ep::Rect) {
        let response = ui.allocate_rect(rect, egui::Sense::click());
        let selected = self.inventory.selected == index;
        let hovered = response.hovered();
        let background = if selected {
            egui::Color32::from_rgb(0x8A, 0x78, 0x3A)
        } else if hovered {
            egui::Color32::from_rgb(0x5E, 0x5E, 0x5E)
        } else {
            egui::Color32::from_rgb(0x3A, 0x3A, 0x3A)
        };
        let border = if selected {
            egui::Color32::from_rgb(0xFF, 0xE6, 0x7A)
        } else {
            egui::Color32::from_rgb(0x18, 0x18, 0x18)
        };
        let painter = ui.painter();
        painter.rect_filled(rect, 2.0, background);
        painter.rect_stroke(
            rect,
            2.0,
            egui::Stroke::new(if selected { 2.0 } else { 1.0 }, border),
            egui::StrokeKind::Inside,
        );

        if let Some(stack) = self.inventory.slots[index] {
            let icon_rect = rect.shrink(5.0);
            let tile = self
                .atlas
                .tile_id(stack.block, stack.block.tile(Face::Side));
            let (u0, v0, u1, v1) = self.atlas.uv(tile);
            painter.image(
                self.atlas_texture.id(),
                icon_rect,
                ep::Rect::from_min_max(ep::Pos2::new(u0, v0), ep::Pos2::new(u1, v1)),
                egui::Color32::WHITE,
            );
            painter.text(
                rect.right_bottom() - egui::Vec2::new(4.0, 3.0),
                egui::Align2::RIGHT_BOTTOM,
                stack.count.to_string(),
                egui::FontId::proportional(12.0),
                egui::Color32::WHITE,
            );
            if hovered {
                response.clone().on_hover_text(stack.block.def().display);
            }
        }

        if response.clicked() {
            self.inventory.select(index);
        }
    }

    fn draw_debug(ui: &egui::Ui) {
        let fps = ui.ctx().input(|i| {
            let dt = i.stable_dt;
            if dt > 0.0 { 1.0 / dt } else { 0.0 }
        });
        let dbg_text = format!(
            "DEBUG\nFPS: {:.0}\nscale: {:.1}",
            fps,
            ui.ctx().pixels_per_point()
        );
        ui.painter().text(
            ep::Pos2::new(8.0, 8.0),
            egui::Align2::LEFT_TOP,
            dbg_text,
            egui::FontId::new(14.0, egui::FontFamily::Proportional),
            egui::Color32::from_rgb(0xFF, 0xFF, 0x00),
        );
    }
}

fn raycast_block<W: VoxelWorld>(
    world: &mut W,
    origin: DVec3,
    direction: DVec3,
    reach: f64,
) -> Option<BlockHit> {
    if direction.length_squared() == 0.0 || reach <= 0.0 {
        return None;
    }
    let direction = direction.normalize();
    let mut voxel = (
        origin.x.floor() as i64,
        origin.y.floor() as i64,
        origin.z.floor() as i64,
    );
    // Avoid offering a placement position inside a block if a caller starts
    // the ray from inside a solid. Fluids are intentionally traversable so
    // they do not become placement targets.
    if world.block_at(voxel.0, voxel.1, voxel.2).is_solid() {
        return None;
    }

    let coordinates = [origin.x, origin.y, origin.z];
    let directions = [direction.x, direction.y, direction.z];
    let mut step = [0_i64; 3];
    let mut t_max = [f64::INFINITY; 3];
    let mut t_delta = [f64::INFINITY; 3];
    for axis in 0..3 {
        if directions[axis] > 0.0 {
            step[axis] = 1;
            t_max[axis] =
                (voxel_axis(voxel, axis) as f64 + 1.0 - coordinates[axis]) / directions[axis];
            t_delta[axis] = 1.0 / directions[axis];
        } else if directions[axis] < 0.0 {
            step[axis] = -1;
            t_max[axis] = (coordinates[axis] - voxel_axis(voxel, axis) as f64) / -directions[axis];
            t_delta[axis] = 1.0 / -directions[axis];
        }
    }

    loop {
        let axis = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
            0
        } else if t_max[1] <= t_max[2] {
            1
        } else {
            2
        };
        let distance = t_max[axis];
        if distance > reach {
            return None;
        }
        let previous = voxel;
        let next_axis_value = voxel_axis(voxel, axis) + step[axis];
        set_voxel_axis(&mut voxel, axis, next_axis_value);
        t_max[axis] += t_delta[axis];
        // Fluids do not stop the placement ray. This makes a water surface
        // non-targetable while still allowing placement against blocks below
        // or behind the water.
        if world.block_at(voxel.0, voxel.1, voxel.2).is_solid() {
            return Some(BlockHit {
                block: voxel,
                place: previous,
            });
        }
    }
}

fn voxel_axis(voxel: (i64, i64, i64), axis: usize) -> i64 {
    match axis {
        0 => voxel.0,
        1 => voxel.1,
        _ => voxel.2,
    }
}

fn set_voxel_axis(voxel: &mut (i64, i64, i64), axis: usize, value: i64) {
    match axis {
        0 => voxel.0 = value,
        1 => voxel.1 = value,
        _ => voxel.2 = value,
    }
}

fn overlaps_aabb(a: Aabb, b: Aabb) -> bool {
    const EPSILON: f64 = 1.0e-7;
    a.max.x > b.min.x + EPSILON
        && a.min.x < b.max.x - EPSILON
        && a.max.y > b.min.y + EPSILON
        && a.min.y < b.max.y - EPSILON
        && a.max.z > b.min.z + EPSILON
        && a.min.z < b.max.z - EPSILON
}

pub struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    launcher: Option<LauncherUi>,
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            launcher: None,
        }
    }

    pub fn run() {
        let event_loop = EventLoop::new().expect("创建 EventLoop 失败");
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = Self::new();
        event_loop.run_app(&mut app).expect("事件循环失败");
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes().with_title("mc"))
                    .expect("创建窗口失败"),
            );
            let gpu = GpuState::new(window.clone());
            let launcher = LauncherUi::new(window.clone(), &gpu);
            self.window = Some(window);
            self.gpu = Some(gpu);
            self.launcher = Some(launcher);
            self.window
                .as_ref()
                .expect("窗口刚刚创建，应当存在")
                .request_redraw();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event
            && let Some(launcher) = self.launcher.as_mut()
        {
            launcher.mouse_motion(delta);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(response) = self
            .launcher
            .as_mut()
            .map(|launcher| launcher.input(&event))
            && response.repaint
            && let Some(window) = self.window.as_ref()
        {
            window.request_redraw();
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::F3),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if let Some(launcher) = self.launcher.as_mut() {
                    launcher.toggle_debug();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(gpu) = self.gpu.as_mut()
                    && let Some(launcher) = self.launcher.as_mut()
                {
                    launcher.render(gpu, event_loop);
                }
            }
            _ => {}
        }
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use glam::DVec3;

    use super::{
        Block, ColumnGen, VoxelWorld, WorldHeightmap, WorldInfo, install_cjk_font, raycast_block,
        safe_spawn_y,
    };

    #[derive(Default)]
    struct RayWorld {
        blocks: HashSet<(i64, i64, i64)>,
        fluids: HashSet<(i64, i64, i64)>,
    }

    impl VoxelWorld for RayWorld {
        fn block_at(&mut self, x: i64, y: i64, z: i64) -> Block {
            if self.blocks.contains(&(x, y, z)) {
                Block::Stone
            } else if self.fluids.contains(&(x, y, z)) {
                Block::Water
            } else {
                Block::Air
            }
        }
    }

    #[test]
    fn raycast_returns_hit_block_and_adjacent_placement_cell() {
        let mut world = RayWorld {
            blocks: HashSet::from([(3, 1, 0)]),
            ..Default::default()
        };
        let hit = raycast_block(&mut world, DVec3::new(0.5, 1.5, 0.5), DVec3::X, 8.0);
        assert_eq!(
            hit,
            Some(super::BlockHit {
                block: (3, 1, 0),
                place: (2, 1, 0),
            })
        );
    }

    #[test]
    fn raycast_respects_reach() {
        let mut world = RayWorld {
            blocks: HashSet::from([(3, 1, 0)]),
            ..Default::default()
        };
        assert!(raycast_block(&mut world, DVec3::new(0.5, 1.5, 0.5), DVec3::X, 2.49,).is_none());
    }

    #[test]
    fn raycast_passes_through_water_and_places_against_submerged_block() {
        let mut world = RayWorld {
            blocks: HashSet::from([(3, 1, 0)]),
            fluids: HashSet::from([(1, 1, 0), (2, 1, 0)]),
        };
        let hit = raycast_block(&mut world, DVec3::new(0.5, 1.5, 0.5), DVec3::X, 8.0);
        assert_eq!(
            hit,
            Some(super::BlockHit {
                block: (3, 1, 0),
                place: (2, 1, 0),
            })
        );
    }

    #[test]
    fn bundled_font_supports_ui_chinese() {
        let ctx = egui::Context::default();
        install_cjk_font(&ctx);
        let output = ctx.run_ui(Default::default(), |ui| {
            let font_id = egui::FontId::proportional(14.0);
            assert!(ui.fonts_mut(|fonts| {
                fonts.has_glyphs(&font_id, "选择世界 创建并进入 WASD 移动")
            }));
        });
        output.drop_without_applying_deltas();
    }

    #[test]
    fn generated_spawn_is_not_in_water() {
        let info = WorldInfo::generate("test".to_string(), 2026_0904);
        let heightmap = WorldHeightmap::new(info.seed);
        let column_gen = ColumnGen::new(info.seed);
        let height = heightmap.height(info.spawn_x, info.spawn_z);
        let column = column_gen.generate(info.spawn_x, info.spawn_z, height);

        assert_eq!(safe_spawn_y(&column), Some(info.spawn_height));
        assert_ne!((info.spawn_x, info.spawn_z), (0, 0));
    }
}
