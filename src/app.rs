//! 应用入口：`winit 0.30 ApplicationHandler`，窗口在 `resumed()` 里建。
//!
//! 启动 UI：MC 风格主菜单，用 egui 绘制。F3 切换调试模式。
//! 视觉上使用程序化脏土铺底、像素化阴影/倒角、等距草方块 logo 和
//! 鼠标视差，让启动器更接近方块沙盒的材质感。
//! 当前已经接通：首页 → 世界列表 → 创建世界 → 世界预览。

use std::sync::{Arc, mpsc};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use egui::epaint as ep;
use glam::{DVec3, Mat4};
use rayon::prelude::*;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::player::GameMode;
use crate::player::inventory::{HOTBAR_SIZE, INVENTORY_SIZE, Inventory, ItemStack};
use crate::player::physics::{Aabb, Player, PlayerInput};
use crate::render::camera::Camera;
use crate::render::gpu::{GpuDebugStats, GpuState};
use crate::render::voxel::{CHUNK_SIZE, LOD_NEAR_RADIUS, VoxelMesh, VoxelMeshWorker};
use crate::world::atlas::Atlas;
use crate::world::block::{Block, Face};
use crate::world::column::{Column, ColumnGen, Y_MAX, Y_MIN};
use crate::world::continent::WorldHeightmap;
use crate::world::recipe::{self, Recipe, RecipeKind};
use crate::world::voxel::{GeneratedVoxelWorld, VoxelWorld};

/// 默认视距（方块半径）。12 个 MC 区块 = 12 × 16 = 192 格。
/// 近环仍是全精度网格，远环由 [`LOD_NEAR_RADIUS`] 之外的粗高度场 LOD
/// 接管；世界创建时只预生成近环，远环柱由网格 worker 按粗粒度补齐。
///
/// 网格是以玩家为中心的方形区域，玩家最多移动 `MESH_REBUILD_DISTANCE`
/// 格才重建，因此相机远裁剪面必须大于
/// `(MESH_RADIUS + MESH_REBUILD_DISTANCE) * sqrt(2)` 才不会切掉角落；
/// 当前远裁剪面为 384，如需继续上调视距请同步调整
/// `Camera::view_projection`。
const MESH_RADIUS: i64 = 192;
/// 世界创建阶段只需要全精度近环；更远的 LOD 柱会在首次网格请求时生成。
const INITIAL_COLUMNS_RADIUS: i64 = LOD_NEAR_RADIUS;
/// 当前仍是整片网格一起重建，但近环半径降到 64 且远环走粗高度场后，
/// 单次重建不会再全量扫描 192 格范围内的每个方块柱。
const MESH_REBUILD_DISTANCE: i64 = 32;
const BLOCK_REACH: f64 = 8.0;
const JEI_COLUMNS: usize = 6;
const JEI_PANEL_WIDTH: f32 = 230.0;
const JEI_PANEL_HEIGHT: f32 = 465.0;

/// How many recent frame times the F3 sparkline keeps.
const DEBUG_FRAME_HISTORY: usize = 180;
/// Minimum wall time between `/proc` resource samples.
const SYSTEM_SAMPLE_INTERVAL_SECS: f64 = 0.5;

/// A very small rolling timer used by the F3 overlay.
///
/// `avg_ms` is an exponential moving average (alpha 0.1) so short-lived
/// hitches still move the number without making the overlay unreadable.
#[derive(Clone, Copy, Default)]
struct StageTiming {
    last_ms: f64,
    avg_ms: f64,
    max_ms: f64,
}

impl StageTiming {
    fn record(&mut self, seconds: f64) {
        let ms = (seconds * 1000.0).max(0.0);
        self.last_ms = ms;
        self.avg_ms = if self.avg_ms <= 0.0 {
            ms
        } else {
            self.avg_ms * 0.9 + ms * 0.1
        };
        self.max_ms = self.max_ms.max(ms);
    }
}

/// Per-frame CPU stage timings plus counters that explain what the frame was
/// doing. Background mesh build time is recorded separately because it happens
/// on the mesher thread and is the most likely source of long stalls.
#[derive(Default)]
struct FrameProfiler {
    total: StageTiming,
    physics: StageTiming,
    raycast: StageTiming,
    mesh: StageTiming,
    highlight: StageTiming,
    ui: StageTiming,
    tessellate: StageTiming,
    acquire: StageTiming,
    submit: StageTiming,
    present: StageTiming,
    mesh_build: StageTiming,
    frame_history_ms: Vec<f32>,
    physics_ticks: u32,
    ui_shapes: usize,
    clipped_primitives: usize,
    mesh_build_ms: f64,
    mesh_upload_ms: f64,
    mesh_request_latency_ms: f64,
    mesh_pending: usize,
    underwater: bool,
}

impl FrameProfiler {
    fn record_frame_time(&mut self, seconds: f64) {
        self.total.record(seconds);
        let ms = (seconds * 1000.0) as f32;
        if self.frame_history_ms.len() >= DEBUG_FRAME_HISTORY {
            self.frame_history_ms.remove(0);
        }
        self.frame_history_ms.push(ms);
    }

    /// CPU work on the render thread, excluding time blocked in present/GPU.
    fn cpu_active_ms(&self) -> f64 {
        self.physics.last_ms
            + self.raycast.last_ms
            + self.mesh.last_ms
            + self.highlight.last_ms
            + self.ui.last_ms
            + self.tessellate.last_ms
            + self.acquire.last_ms
            + self.submit.last_ms
    }
}

/// Process-wide resource numbers sampled from `/proc` on Linux. Values stay at
/// zero on unsupported platforms instead of guessing.
#[derive(Clone, Copy, Default)]
#[allow(dead_code)] // 系统采样字段已接好，等 debug HUD 展示后再移除 allow。
struct SystemStats {
    rss_bytes: u64,
    peak_rss_bytes: u64,
    cpu_percent: f32,
    thread_count: u32,
    load_avg_1: f32,
}

struct SystemSampler {
    last_sample: Instant,
    last_cpu_ticks: Option<u64>,
}

impl SystemSampler {
    fn new() -> Self {
        Self {
            last_sample: Instant::now(),
            last_cpu_ticks: proc_cpu_ticks(),
        }
    }

    fn due(&self) -> bool {
        self.last_sample.elapsed().as_secs_f64() >= SYSTEM_SAMPLE_INTERVAL_SECS
    }

    fn sample(&mut self) -> SystemStats {
        let elapsed = self.last_sample.elapsed().as_secs_f64().max(0.001);
        let now_ticks = proc_cpu_ticks();
        let cpu_percent = match (self.last_cpu_ticks, now_ticks) {
            (Some(previous), Some(now)) => {
                // Linux reports utime+stime in USER_HZ (100 Hz on essentially
                // every desktop kernel). delta_ticks / elapsed is then the
                // percentage of one core, which may exceed 100% on multithread.
                (now.saturating_sub(previous) as f64 / elapsed) as f32
            }
            _ => 0.0,
        };
        self.last_cpu_ticks = now_ticks;
        self.last_sample = Instant::now();
        let (rss_bytes, peak_rss_bytes, thread_count) = read_proc_status();
        SystemStats {
            rss_bytes,
            peak_rss_bytes,
            cpu_percent,
            thread_count,
            load_avg_1: read_load_avg(),
        }
    }
}

#[cfg(target_os = "linux")]
fn read_proc_status() -> (u64, u64, u32) {
    let mut rss = 0;
    let mut peak = 0;
    let mut threads = 0;
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            let mut parts = line.split_whitespace();
            match parts.next() {
                Some("VmRSS:") => {
                    rss = parts
                        .next()
                        .and_then(|value| value.parse::<u64>().ok())
                        .unwrap_or(0)
                        * 1024;
                }
                Some("VmHWM:") => {
                    peak = parts
                        .next()
                        .and_then(|value| value.parse::<u64>().ok())
                        .unwrap_or(0)
                        * 1024;
                }
                Some("Threads:") => {
                    threads = parts
                        .next()
                        .and_then(|value| value.parse::<u32>().ok())
                        .unwrap_or(0);
                }
                _ => {}
            }
        }
    }
    (rss, peak, threads)
}

#[cfg(not(target_os = "linux"))]
fn read_proc_status() -> (u64, u64, u32) {
    (0, 0, 0)
}

#[cfg(target_os = "linux")]
fn read_load_avg() -> f32 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|contents| {
            contents
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<f32>().ok())
        })
        .unwrap_or(0.0)
}

#[cfg(not(target_os = "linux"))]
fn read_load_avg() -> f32 {
    0.0
}

#[cfg(target_os = "linux")]
fn proc_cpu_ticks() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Field 2 (`comm`) can contain spaces and parentheses, so parse from the
    // last `)`. utime/stime are then fields 14/15 overall (indices 11/12 here).
    let after_comm = &stat[stat.rfind(')')? + 1..];
    let fields: Vec<&str> = after_comm.split_whitespace().collect();
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some(utime + stime)
}

#[cfg(not(target_os = "linux"))]
fn proc_cpu_ticks() -> Option<u64> {
    None
}

/// Generate a fresh seed for the world-creation form without adding a
/// runtime dependency. The system clock provides enough entropy for user
/// initiated world creation; the resulting value is still displayed and can
/// be copied to reproduce the same world later.
fn random_world_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos() as u64);
    // Mix the timestamp bits so nearby creation times do not produce seeds
    // with an obvious decimal pattern.
    let mut value = nanos.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

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
    mesh_origin: DVec3,
    camera_position: DVec3,
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
            .map(|(x, z)| column_gen.generate_with_terrain(x, z, heightmap.sample(x, z)))
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

/// 在世界原点附近寻找可安全站立的出生点。
///
/// 按 128 格间距稀疏扫描，避免在大海中逐格搜索。沙滩、山地和
/// 其他非流体地表都可以成为出生点；海陆分布由高度图本身决定。
fn find_safe_spawn_with_progress(
    heightmap: &WorldHeightmap,
    column_gen: &ColumnGen,
    mut on_progress: impl FnMut(f32),
) -> (i64, i64, i64) {
    const SEARCH_STEP: i64 = 128;
    const MAX_SEARCH_RADIUS: i64 = 32_768;
    const MAX_RING: i64 = MAX_SEARCH_RADIUS / SEARCH_STEP;

    for ring_radius in 0..=MAX_RING {
        if ring_radius == 0 || ring_radius % 4 == 0 {
            on_progress(ring_radius as f32 / MAX_RING as f32);
        }
        let ring: Vec<_> = if ring_radius == 0 {
            vec![(0, 0)]
        } else {
            let mut ring = Vec::with_capacity((ring_radius * 8) as usize);
            for x in -ring_radius..=ring_radius {
                ring.push((x * SEARCH_STEP, -ring_radius * SEARCH_STEP));
                ring.push((x * SEARCH_STEP, ring_radius * SEARCH_STEP));
            }
            for z in (-ring_radius + 1)..ring_radius {
                ring.push((-ring_radius * SEARCH_STEP, z * SEARCH_STEP));
                ring.push((ring_radius * SEARCH_STEP, z * SEARCH_STEP));
            }
            ring
        };
        if let Some((_, _, x, z, y)) = ring
            .into_par_iter()
            .enumerate()
            .filter_map(|(order, (x, z))| {
                spawn_candidate(heightmap, column_gen, x, z)
                    .map(|(distance, x, z, y)| (distance, order, x, z, y))
            })
            .min_by_key(|candidate| (candidate.0, candidate.1))
        {
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
    let terrain = heightmap.sample(x, z);
    let column = column_gen.generate_with_terrain(x, z, terrain);
    let spawn_y = safe_spawn_y(&column)?;
    Some((x * x + z * z, x, z, spawn_y))
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
    profiler: FrameProfiler,
    system_stats: SystemStats,
    system_sampler: SystemSampler,
    gpu_debug: GpuDebugStats,
    #[allow(dead_code)] // 预留给 F3 / debug HUD 的 GPU 信息展示。
    gpu_info: String,
    mesh_request_started: Option<Instant>,
    target_block_kind: Option<Block>,
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
    game_mode: GameMode,
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
    /// 当前鼠标光标拿起的物品堆；这是物品栏 UI 状态，不属于世界存档。
    cursor_stack: Option<ItemStack>,
    cursor_source: Option<usize>,
    /// JEI 浏览器只在打开物品栏时显示，J 可以切换它的可见性。
    jei_visible: bool,
    jei_page: usize,
    jei_search: String,
    jei_selected: Option<Block>,
    atlas: Arc<Atlas>,
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
            profiler: FrameProfiler::default(),
            system_stats: SystemStats::default(),
            system_sampler: SystemSampler::new(),
            gpu_debug: gpu.debug_stats(),
            gpu_info: {
                let info = &gpu.adapter_info;
                format!(
                    "{} · {:?}/{:?} · {}",
                    info.name, info.backend, info.device_type, info.driver
                )
            },
            mesh_request_started: None,
            target_block_kind: None,
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
            draft_seed: random_world_seed().to_string(),
            form_error: None,
            exit_requested: false,
            physics_world: None,
            player: None,
            game_mode: GameMode::Survival,
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
            cursor_stack: None,
            cursor_source: None,
            jei_visible: false,
            jei_page: 0,
            jei_search: String::new(),
            jei_selected: None,
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
        let raw_elapsed = now.duration_since(self.last_frame).as_secs_f64();
        let frame_dt = raw_elapsed.min(0.25);
        self.last_frame = now;
        self.profiler.record_frame_time(raw_elapsed);

        if self.system_sampler.due() {
            self.system_stats = self.system_sampler.sample();
        }

        let timer = Instant::now();
        self.advance_physics(frame_dt);
        self.profiler.physics.record(timer.elapsed().as_secs_f64());

        self.time += frame_dt;
        self.poll_world_creation();
        self.animate_creation_progress(frame_dt);

        let timer = Instant::now();
        self.update_target_block();
        self.profiler.raycast.record(timer.elapsed().as_secs_f64());

        let timer = Instant::now();
        self.ensure_world_mesh(gpu);
        self.profiler.mesh.record(timer.elapsed().as_secs_f64());

        let timer = Instant::now();
        gpu.upload_highlight(
            self.target_block.map(|target| target.block),
            self.mesh_origin,
        );
        self.profiler
            .highlight
            .record(timer.elapsed().as_secs_f64());

        let in_world = self.screen == LauncherScreen::InWorld;
        let underwater = in_world && self.camera_is_underwater();
        self.profiler.underwater = underwater;

        let raw_input = self.winit.take_egui_input(&self.window);
        // 克隆 Context，避免在 run_ui 的借用期间同时借用 LauncherUi 的 ctx 字段。
        let ctx = self.ctx.clone();
        let timer = Instant::now();
        let mut full_output = ctx.run_ui(raw_input, |ui| {
            self.build(ui);
        });
        self.profiler.ui.record(timer.elapsed().as_secs_f64());
        self.profiler.ui_shapes = full_output.shapes.len();
        if self.exit_requested {
            full_output.textures_delta.clear();
            event_loop.exit();
            return;
        }
        self.winit
            .handle_platform_output(&self.window, full_output.platform_output);

        let timer = Instant::now();
        let clipped = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        self.profiler
            .tessellate
            .record(timer.elapsed().as_secs_f64());
        self.profiler.clipped_primitives = clipped.len();

        let mut textures_delta = std::mem::take(&mut full_output.textures_delta);
        for (id, deltas) in &textures_delta.set {
            for delta in deltas {
                self.renderer
                    .update_texture(&gpu.device, &gpu.queue, *id, delta);
            }
        }
        let timer = Instant::now();
        let output = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                self.profiler.acquire.record(timer.elapsed().as_secs_f64());
                textures_delta.clear();
                return;
            }
        };
        self.profiler.acquire.record(timer.elapsed().as_secs_f64());
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sd = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gpu.config.width, gpu.config.height],
            pixels_per_point: ctx.pixels_per_point(),
        };
        let camera_matrix = self.camera_matrix(gpu);
        let render_state = WorldRenderState {
            in_world,
            underwater,
            view_proj: camera_matrix,
            mesh_origin: self.mesh_origin,
            camera_position: self
                .player
                .as_ref()
                .map(|player| self.camera.eye_position(player) - self.mesh_origin)
                .unwrap_or(DVec3::ZERO),
        };
        let timer = Instant::now();
        Self::submit_frame(gpu, &view, &clipped, &sd, &mut self.renderer, render_state);
        self.profiler.submit.record(timer.elapsed().as_secs_f64());

        let timer = Instant::now();
        gpu.queue.present(output);
        self.profiler.present.record(timer.elapsed().as_secs_f64());

        self.gpu_debug = gpu.debug_stats();

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
        self.game_mode = GameMode::Survival;
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
        self.mesh_request_started = None;
        self.target_block = None;
        self.target_block_kind = None;
        self.profiler = FrameProfiler::default();
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
        self.mesh_request_started = None;
        self.target_block = None;
        self.target_block_kind = None;
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
        if self.inventory.open && self.ctx.egui_wants_keyboard_input() && code != KeyCode::Escape {
            return;
        }
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
        if code == KeyCode::KeyJ && pressed && !event.repeat && self.inventory.open {
            self.jei_visible = !self.jei_visible;
            return;
        }
        if code == KeyCode::F4 && pressed && !event.repeat && !self.inventory.open {
            self.toggle_game_mode();
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
            self.target_block_kind = None;
            return;
        }
        let (Some(player), Some(world)) = (self.player.as_ref(), self.physics_world.as_mut())
        else {
            self.target_block = None;
            self.target_block_kind = None;
            return;
        };
        let hit = raycast_block(
            world,
            self.camera.eye_position(player),
            self.camera.forward(),
            BLOCK_REACH,
        );
        self.target_block = hit;
        self.target_block_kind =
            hit.map(|target| world.block_at(target.block.0, target.block.1, target.block.2));
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
        if block == Block::Air
            || block.is_fluid()
            || (!self.game_mode.is_creative() && block.def().hardness.is_none())
        {
            return;
        }
        world.remove_block_with_fluid_update(target.block.0, target.block.1, target.block.2);
        if !self.game_mode.is_creative() {
            self.inventory.give(block);
        }
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
        if !self.game_mode.is_creative() {
            self.inventory.take_selected();
        }
        self.mesh_needs_rebuild = true;
        self.mesh_requested_center = None;
    }

    fn toggle_inventory(&mut self) {
        if self.inventory.open {
            self.close_inventory();
            return;
        }
        self.inventory.toggle();
        self.jei_visible = self.game_mode.is_creative();
        self.stop_player_input();
        let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        self.window.set_cursor_visible(true);
    }

    fn close_inventory(&mut self) {
        self.return_cursor_stack();
        self.inventory.close();
        let _ = self
            .window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined));
        self.window.set_cursor_visible(false);
    }

    fn return_cursor_stack(&mut self) {
        if self.cursor_stack.is_some() {
            self.cursor_stack = self.inventory.return_stack(self.cursor_stack.take());
        }
        self.cursor_source = None;
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

    fn toggle_game_mode(&mut self) {
        self.game_mode = self.game_mode.toggle();
        if self.game_mode.is_creative() {
            self.inventory.refill_hotbar();
        }
        if let Some(player) = self.player.as_mut() {
            player.velocity = DVec3::ZERO;
            player.on_ground = false;
        }
        self.stop_player_input();
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
            creative: self.game_mode.is_creative(),
        }
    }

    /// 以 60 TPS 固定步长推进物理，渲染帧率变化不会改变移动和重力结果。
    fn advance_physics(&mut self, frame_dt: f64) {
        if self.screen != LauncherScreen::InWorld || self.inventory.open {
            self.profiler.physics_ticks = 0;
            return;
        }

        const FIXED_DT: f64 = 1.0 / 60.0;
        self.physics_accumulator = (self.physics_accumulator + frame_dt).min(0.25);
        let mut ticks = 0;
        let mut world_changed = false;
        while self.physics_accumulator >= FIXED_DT {
            let input = self.player_input(self.jump_requested && ticks == 0);
            let (Some(world), Some(player)) = (self.physics_world.as_mut(), self.player.as_mut())
            else {
                break;
            };
            world_changed |= world.tick_gravity();
            player.step(world, input, FIXED_DT);
            self.physics_accumulator -= FIXED_DT;
            ticks += 1;
        }
        if ticks > 0 {
            self.jump_requested = false;
        }
        self.profiler.physics_ticks = ticks;
        if world_changed {
            self.mesh_needs_rebuild = true;
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
                gpu.atlas.clone(),
                initial_columns,
            ));
        }

        if let Some(worker) = self.mesh_worker.as_mut()
            && let Some(result) = worker.try_take_latest()
        {
            let upload_start = Instant::now();
            gpu.upload_mesh(&result.mesh);
            if let Some(probes) = result.probes.as_ref() {
                gpu.upload_probes(probes);
            }
            self.profiler.mesh_upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;
            self.profiler
                .mesh_build
                .record(result.build_time.as_secs_f64());
            self.profiler.mesh_build_ms = result.build_time.as_secs_f64() * 1000.0;
            if let Some(started) = self.mesh_request_started.take() {
                self.profiler.mesh_request_latency_ms = started.elapsed().as_secs_f64() * 1000.0;
            }
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
            self.mesh_request_started = Some(Instant::now());
            self.mesh_requested_center = Some(center);
            self.mesh_needs_rebuild = false;
        }
        if let Some(worker) = self.mesh_worker.as_ref() {
            self.profiler.mesh_pending = worker.pending_requests();
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
            mesh_origin,
            camera_position,
        } = render_state;
        if in_world {
            gpu.update_camera(view_proj, underwater, mesh_origin, camera_position);
        }
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mc frame"),
            });
        if in_world {
            gpu.draw_shadows(&mut enc);
        }
        // Pass 1: render the sky and opaque world into a sampleable scene
        // texture. Water needs this backdrop for Photon-style refraction.
        {
            let p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(if in_world { "mc world" } else { "mc clear" }),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: if in_world { gpu.scene_view() } else { view },
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // 世界背景由 analytic sky pass 绘制；这里先清黑，
                        // 菜单仍保留原来的深色清屏。
                        load: wgpu::LoadOp::Clear(if in_world {
                            wgpu::Color::BLACK
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
                gpu.draw_sky(&mut p);
                gpu.draw_voxels(&mut p);
            }
        }
        if in_world {
            // Pass 2: copy the opaque scene to the swapchain.
            {
                let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mc scene composite"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                gpu.draw_scene(&mut p);
                // Additive screen-space scattering gives underwater light
                // shafts and sunbeams filtered through leaves.
                gpu.draw_god_rays(&mut p);
            }

            // Pass 3: transparent geometry samples the opaque scene. Reusing
            // the opaque depth buffer keeps water behind cliffs and shores.
            {
                let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mc water and transparent"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: gpu.depth_view(),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                gpu.draw_transparent_voxels(&mut p);
                gpu.draw_highlight(&mut p);
            }
        }
        // Pass 4: egui overlay
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
            self.draw_backdrop(ui, screen);
        }

        match self.screen {
            LauncherScreen::Home => self.draw_home(ui, screen),
            LauncherScreen::Worlds => self.draw_worlds(ui, screen),
            LauncherScreen::CreateWorld => self.draw_create_world(ui, screen),
            LauncherScreen::CreatingWorld => self.draw_creating_world(ui, screen),
            LauncherScreen::InWorld => self.draw_in_world(ui, screen),
        }

        if self.debug {
            self.draw_debug(ui);
        }
    }

    fn color_lerp(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
        let t = t.clamp(0.0, 1.0);
        let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
        egui::Color32::from_rgba_unmultiplied(
            mix(a.r(), b.r()),
            mix(a.g(), b.g()),
            mix(a.b(), b.b()),
            mix(a.a(), b.a()),
        )
    }

    fn hash01(mut value: u64) -> f32 {
        value ^= value >> 33;
        value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
        value ^= value >> 33;
        value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        value ^= value >> 33;
        ((value >> 11) as f64 / ((1u64 << 53) as f64)) as f32
    }

    fn vertical_gradient(
        painter: &egui::Painter,
        rect: ep::Rect,
        top: egui::Color32,
        bottom: egui::Color32,
    ) {
        let mut mesh = egui::Mesh::default();
        mesh.colored_vertex(rect.left_top(), top);
        mesh.colored_vertex(rect.right_top(), top);
        mesh.colored_vertex(rect.left_bottom(), bottom);
        mesh.colored_vertex(rect.right_bottom(), bottom);
        mesh.add_triangle(0, 1, 2);
        mesh.add_triangle(2, 1, 3);
        painter.add(egui::Shape::mesh(mesh));
    }

    fn horizontal_gradient(
        painter: &egui::Painter,
        rect: ep::Rect,
        left: egui::Color32,
        right: egui::Color32,
    ) {
        let mut mesh = egui::Mesh::default();
        mesh.colored_vertex(rect.left_top(), left);
        mesh.colored_vertex(rect.right_top(), right);
        mesh.colored_vertex(rect.left_bottom(), left);
        mesh.colored_vertex(rect.right_bottom(), right);
        mesh.add_triangle(0, 2, 1);
        mesh.add_triangle(1, 2, 3);
        painter.add(egui::Shape::mesh(mesh));
    }

    fn paint_vignette(painter: &egui::Painter, screen: ep::Rect) {
        let transparent = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 0);
        let edge = egui::Color32::from_rgba_unmultiplied(0, 0, 0, 150);
        let band_y = (screen.height() * 0.20).clamp(58.0, 170.0);
        let band_x = (screen.width() * 0.12).clamp(48.0, 150.0);

        let top = ep::Rect::from_min_max(
            screen.min,
            ep::Pos2::new(screen.max.x, screen.min.y + band_y),
        );
        let bottom = ep::Rect::from_min_max(
            ep::Pos2::new(screen.min.x, screen.max.y - band_y),
            screen.max,
        );
        let left = ep::Rect::from_min_max(
            screen.min,
            ep::Pos2::new(screen.min.x + band_x, screen.max.y),
        );
        let right = ep::Rect::from_min_max(
            ep::Pos2::new(screen.max.x - band_x, screen.min.y),
            screen.max,
        );

        Self::vertical_gradient(painter, top, edge, transparent);
        Self::vertical_gradient(painter, bottom, transparent, edge);
        Self::horizontal_gradient(painter, left, edge, transparent);
        Self::horizontal_gradient(painter, right, transparent, edge);
    }

    fn draw_backdrop(&self, ui: &egui::Ui, screen: ep::Rect) {
        let painter = ui.painter();
        Self::vertical_gradient(
            painter,
            screen,
            egui::Color32::from_rgb(0x1B, 0x2B, 0x34),
            egui::Color32::from_rgb(0x0B, 0x10, 0x13),
        );

        let texture_id = self.atlas_texture.id();
        let tile_index = self
            .atlas
            .tile_index("dirt.png")
            .or_else(|| self.atlas.tile_index("stone.png"))
            .unwrap_or(0);
        let (u0, v0, u1, v1) = self.atlas.uv(tile_index);
        let uv = ep::Rect::from_min_max(ep::Pos2::new(u0, v0), ep::Pos2::new(u1, v1));

        let tile = (screen.width().min(screen.height()) * 0.13).clamp(56.0, 116.0);
        let pointer = ui.ctx().pointer_latest_pos().unwrap_or(screen.center());
        let parallax_x =
            ((pointer.x - screen.center().x) / screen.width().max(1.0)).clamp(-1.0, 1.0);
        let parallax_y =
            ((pointer.y - screen.center().y) / screen.height().max(1.0)).clamp(-1.0, 1.0);
        let drift_x = (self.time as f32 * 5.0) % tile;
        let drift_y = (self.time as f32 * 2.5) % tile;
        let offset = ep::Vec2::new(parallax_x * 16.0 + drift_x, parallax_y * 10.0 + drift_y);

        let cols = (screen.width() / tile).ceil() as i32 + 3;
        let rows = (screen.height() / tile).ceil() as i32 + 3;
        for row in -1..rows {
            for col in -1..cols {
                let x = screen.min.x + col as f32 * tile - offset.x;
                let y = screen.min.y + row as f32 * tile - offset.y;
                let rect = ep::Rect::from_min_size(ep::Pos2::new(x, y), ep::Vec2::splat(tile));
                let tint = if (row + col) & 1 == 0 {
                    egui::Color32::from_rgba_unmultiplied(92, 78, 60, 27)
                } else {
                    egui::Color32::from_rgba_unmultiplied(52, 62, 55, 25)
                };
                painter.image(texture_id, rect, uv, tint);
            }
        }

        painter.rect_filled(
            screen,
            0.0,
            egui::Color32::from_rgba_unmultiplied(5, 8, 11, 118),
        );
        Self::vertical_gradient(
            painter,
            screen,
            egui::Color32::from_rgba_unmultiplied(30, 45, 58, 48),
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 92),
        );

        let particle_count =
            ((screen.width() * screen.height()) / 26_000.0).clamp(22.0, 76.0) as usize;
        for i in 0..particle_count {
            let seed = i as u64 * 0x9E37_79B9_7F4A_7C15;
            let x = Self::hash01(seed) * screen.width();
            let speed = 5.0 + Self::hash01(seed ^ 0xDEAD_BEEF) * 18.0;
            let y = (Self::hash01(seed ^ 0xBEEF_CAFE) * screen.height() + self.time as f32 * speed)
                % screen.height();
            let radius = 0.7 + Self::hash01(seed ^ 0xF00D_BABE) * 1.7;
            let alpha = 18 + (Self::hash01(seed ^ 0x1234_5678) * 42.0) as u8;
            painter.circle_filled(
                ep::Pos2::new(screen.min.x + x, screen.min.y + y),
                radius,
                egui::Color32::from_rgba_unmultiplied(235, 226, 194, alpha),
            );
        }

        Self::paint_vignette(painter, screen);
    }

    fn panel(ui: &egui::Ui, rect: ep::Rect) {
        let painter = ui.painter();
        let shadow = ep::Rect::from_min_size(rect.min + egui::Vec2::new(5.0, 7.0), rect.size());
        painter.rect_filled(
            shadow,
            4.0,
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 125),
        );
        painter.rect_filled(
            rect,
            4.0,
            egui::Color32::from_rgba_unmultiplied(20, 16, 12, 226),
        );
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(0x5B, 0x40, 0x26)),
            egui::StrokeKind::Inside,
        );
        painter.rect_stroke(
            rect.shrink(3.0),
            2.0,
            egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 28),
            ),
            egui::StrokeKind::Inside,
        );
        painter.line_segment(
            [
                rect.left_top() + egui::Vec2::new(8.0, 5.0),
                rect.right_top() + egui::Vec2::new(-8.0, 5.0),
            ],
            egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 35),
            ),
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

    fn draw_badge(ui: &egui::Ui, rect: ep::Rect, label: &str) {
        let painter = ui.painter();
        painter.rect_filled(
            rect,
            3.0,
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 110),
        );
        painter.rect_stroke(
            rect,
            3.0,
            egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 42),
            ),
            egui::StrokeKind::Inside,
        );
        Self::text(
            ui,
            rect.center(),
            label,
            11.0,
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 170),
        );
    }

    fn draw_footer(ui: &egui::Ui, screen: ep::Rect, left: &str, right: &str) {
        let bar_h = 30.0;
        let bar = ep::Rect::from_min_max(
            ep::Pos2::new(screen.min.x, screen.max.y - bar_h),
            screen.max,
        );
        let painter = ui.painter();
        painter.rect_filled(
            bar,
            0.0,
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 132),
        );
        painter.rect_filled(
            ep::Rect::from_min_max(bar.min, ep::Pos2::new(bar.max.x, bar.min.y + 1.0)),
            0.0,
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 22),
        );
        Self::left_text(
            ui,
            ep::Pos2::new(bar.min.x + 12.0, bar.center().y),
            left,
            11.0,
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 145),
        );
        painter.text(
            ep::Pos2::new(bar.max.x - 12.0, bar.center().y),
            egui::Align2::RIGHT_CENTER,
            right.to_string(),
            egui::FontId::new(11.0, egui::FontFamily::Proportional),
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 145),
        );
    }

    fn menu_button(ui: &mut egui::Ui, rect: ep::Rect, label: &str, enabled: bool) -> bool {
        let response = ui.allocate_rect(rect, egui::Sense::click());
        let hovered = enabled && response.hovered();
        let hover_t = ui.ctx().animate_bool_with_time(response.id, hovered, 0.10);
        let pressed = enabled && response.is_pointer_button_down_on();
        let y_shift = if pressed { 2.0 } else { 0.0 };
        let draw_rect = rect.translate(egui::Vec2::new(0.0, y_shift));
        let painter = ui.painter();

        let normal_top = egui::Color32::from_rgb(0x6F, 0x6F, 0x70);
        let normal_bottom = egui::Color32::from_rgb(0x3D, 0x3D, 0x40);
        let hover_top = egui::Color32::from_rgb(0x87, 0x82, 0xC0);
        let hover_bottom = egui::Color32::from_rgb(0x44, 0x3E, 0x75);
        let disabled_top = egui::Color32::from_rgb(0x47, 0x47, 0x47);
        let disabled_bottom = egui::Color32::from_rgb(0x2A, 0x2A, 0x2A);

        let (top, bottom, border, text_color) = if !enabled {
            (
                disabled_top,
                disabled_bottom,
                egui::Color32::from_rgb(0x23, 0x23, 0x23),
                egui::Color32::from_rgb(0x86, 0x86, 0x86),
            )
        } else {
            (
                Self::color_lerp(normal_top, hover_top, hover_t),
                Self::color_lerp(normal_bottom, hover_bottom, hover_t),
                Self::color_lerp(
                    egui::Color32::from_rgb(0x25, 0x25, 0x27),
                    egui::Color32::from_rgb(0xB8, 0xB0, 0xEA),
                    hover_t,
                ),
                Self::color_lerp(
                    egui::Color32::from_rgb(0xE8, 0xE8, 0xE8),
                    egui::Color32::from_rgb(0xFF, 0xF1, 0xA8),
                    hover_t,
                ),
            )
        };

        let shadow_shift = if pressed {
            2.0
        } else if enabled {
            4.0
        } else {
            2.0
        };
        let shadow =
            ep::Rect::from_min_size(rect.min + egui::Vec2::new(0.0, shadow_shift), rect.size());
        painter.rect_filled(
            shadow,
            3.0,
            egui::Color32::from_rgba_unmultiplied(
                0,
                0,
                0,
                if enabled {
                    (118.0 - hover_t * 30.0) as u8
                } else {
                    74
                },
            ),
        );

        Self::vertical_gradient(painter, draw_rect, top, bottom);
        painter.rect_stroke(
            draw_rect,
            3.0,
            egui::Stroke::new(2.0, border),
            egui::StrokeKind::Inside,
        );

        let bevel =
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, if enabled { 42 } else { 16 });
        painter.line_segment(
            [
                draw_rect.left_top() + egui::Vec2::new(6.0, 3.0),
                draw_rect.right_top() + egui::Vec2::new(-6.0, 3.0),
            ],
            egui::Stroke::new(1.0, bevel),
        );
        painter.line_segment(
            [
                draw_rect.left_top() + egui::Vec2::new(3.0, 6.0),
                draw_rect.left_bottom() + egui::Vec2::new(3.0, -6.0),
            ],
            egui::Stroke::new(1.0, bevel),
        );
        painter.line_segment(
            [
                draw_rect.left_bottom() + egui::Vec2::new(6.0, -3.0),
                draw_rect.right_bottom() + egui::Vec2::new(-6.0, -3.0),
            ],
            egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, if enabled { 92 } else { 58 }),
            ),
        );

        if hovered {
            let accent = ep::Rect::from_min_size(
                draw_rect.min + egui::Vec2::new(5.0, 7.0),
                egui::Vec2::new(3.0, draw_rect.height() - 14.0),
            );
            painter.rect_filled(
                accent,
                1.0,
                egui::Color32::from_rgba_unmultiplied(255, 218, 112, (210.0 * hover_t) as u8),
            );
        }

        painter.text(
            draw_rect.center() + egui::Vec2::new(0.0, 2.0),
            egui::Align2::CENTER_CENTER,
            label.to_string(),
            egui::FontId::new(17.0, egui::FontFamily::Proportional),
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 155),
        );
        painter.text(
            draw_rect.center(),
            egui::Align2::CENTER_CENTER,
            label.to_string(),
            egui::FontId::new(17.0, egui::FontFamily::Proportional),
            text_color,
        );

        response.clicked() && enabled
    }

    fn draw_voxel_cube(ui: &egui::Ui, center: ep::Pos2, half: f32, depth: f32, time: f64) {
        let painter = ui.painter();
        let c = center + egui::Vec2::new(0.0, (time as f32 * 1.6).sin() * 2.0);
        let top = vec![
            c + egui::Vec2::new(-half, 0.0),
            c + egui::Vec2::new(0.0, -half * 0.55),
            c + egui::Vec2::new(half, 0.0),
            c + egui::Vec2::new(0.0, half * 0.55),
        ];
        let left = vec![
            c + egui::Vec2::new(-half, 0.0),
            c + egui::Vec2::new(0.0, half * 0.55),
            c + egui::Vec2::new(0.0, half * 0.55 + depth),
            c + egui::Vec2::new(-half, depth),
        ];
        let right = vec![
            c + egui::Vec2::new(half, 0.0),
            c + egui::Vec2::new(0.0, half * 0.55),
            c + egui::Vec2::new(0.0, half * 0.55 + depth),
            c + egui::Vec2::new(half, depth),
        ];

        painter.rect_filled(
            ep::Rect::from_center_size(
                c + egui::Vec2::new(0.0, depth + half * 0.35),
                egui::Vec2::new(half * 2.4, half * 0.45),
            ),
            4.0,
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 85),
        );
        painter.add(egui::Shape::convex_polygon(
            left,
            egui::Color32::from_rgb(0x6A, 0x42, 0x24),
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0x35, 0x20, 0x10)),
        ));
        painter.add(egui::Shape::convex_polygon(
            right,
            egui::Color32::from_rgb(0x8A, 0x5B, 0x32),
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0x3E, 0x26, 0x13)),
        ));
        painter.add(egui::Shape::convex_polygon(
            top.clone(),
            egui::Color32::from_rgb(0x67, 0xAD, 0x3F),
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0x35, 0x66, 0x22)),
        ));

        painter.line_segment(
            [top[0], top[1]],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(0xA5, 0xDC, 0x72)),
        );
        painter.line_segment(
            [top[1], top[2]],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(0x86, 0xC9, 0x58)),
        );
        painter.line_segment(
            [
                c + egui::Vec2::new(0.0, 0.0),
                c + egui::Vec2::new(0.0, half * 0.55),
            ],
            egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 90)),
        );
        painter.line_segment(
            [
                c + egui::Vec2::new(0.0, half * 0.55),
                c + egui::Vec2::new(0.0, half * 0.55 + depth),
            ],
            egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 110)),
        );
    }

    fn draw_logo(ui: &egui::Ui, center: ep::Pos2, time: f64) {
        let painter = ui.painter();
        Self::draw_voxel_cube(
            ui,
            ep::Pos2::new(center.x - 94.0, center.y + 12.0),
            23.0,
            31.0,
            time,
        );

        let font = egui::FontId::new(84.0, egui::FontFamily::Proportional);
        let pos = ep::Pos2::new(center.x + 18.0, center.y);
        for (dx, dy) in [
            (-3.0, 0.0),
            (3.0, 0.0),
            (0.0, -3.0),
            (0.0, 3.0),
            (-2.0, 2.0),
            (2.0, 2.0),
        ] {
            painter.text(
                pos + egui::Vec2::new(dx, dy),
                egui::Align2::CENTER_CENTER,
                "mc",
                font.clone(),
                egui::Color32::from_rgb(0x38, 0x20, 0x0C),
            );
        }
        painter.text(
            pos + egui::Vec2::new(0.0, 5.0),
            egui::Align2::CENTER_CENTER,
            "mc",
            font.clone(),
            egui::Color32::from_rgb(0x2C, 0x18, 0x07),
        );
        painter.text(
            pos,
            egui::Align2::CENTER_CENTER,
            "mc",
            font.clone(),
            egui::Color32::from_rgb(0xD7, 0xA2, 0x3B),
        );
        painter.text(
            pos - egui::Vec2::new(1.5, 1.5),
            egui::Align2::CENTER_CENTER,
            "mc",
            font,
            egui::Color32::from_rgba_unmultiplied(255, 238, 170, 125),
        );
    }

    fn draw_home(&mut self, ui: &mut egui::Ui, screen: ep::Rect) {
        let w = screen.width();
        let h = screen.height();
        let cx = screen.center().x;

        let badge_w = if w > 720.0 { 166.0 } else { 138.0 };
        Self::draw_badge(
            ui,
            ep::Rect::from_min_size(
                ep::Pos2::new(screen.max.x - badge_w - 16.0, screen.min.y + 16.0),
                egui::Vec2::new(badge_w, 27.0),
            ),
            if w > 720.0 {
                "SNAPSHOT · 0.1.0"
            } else {
                "0.1.0"
            },
        );

        let title_y = (h * 0.22).max(116.0);
        Self::draw_logo(ui, ep::Pos2::new(cx, title_y), self.time);
        Self::text(
            ui,
            ep::Pos2::new(cx, title_y + 72.0),
            "MINECRAFT-LIKE VOXEL SANDBOX",
            12.0,
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 138),
        );

        const SPLASHES: [&str; 7] = [
            "也是方块！",
            "100% Rust!",
            "wgpu powered!",
            "程序化世界！",
            "挖三填一…",
            "正在烤面包…",
            "当心苦力怕！",
        ];
        let splash_index = ((self.time / 3.2) as usize) % SPLASHES.len();
        let splash_bob = (self.time * 3.0).sin() as f32 * 3.0;
        let splash_x = if w >= 760.0 { cx + 174.0 } else { cx };
        let splash_y = title_y + if w >= 760.0 { -26.0 } else { 52.0 };
        Self::text(
            ui,
            ep::Pos2::new(splash_x, splash_y + splash_bob),
            SPLASHES[splash_index],
            15.0,
            egui::Color32::from_rgb(0xFF, 0xEF, 0x62),
        );

        let bw = (w * 0.42).clamp(260.0, 400.0);
        let bh = 42.0;
        let gap = 9.0;
        let bx = cx - bw / 2.0;
        let by0 = (h * 0.47).max(title_y + 118.0);
        let labels = ["单人游戏", "多人游戏", "选项…", "退出游戏"];
        let enabled = [true, false, false, true];

        for (i, label) in labels.iter().enumerate() {
            let rect = ep::Rect::from_min_size(
                ep::Pos2::new(bx, by0 + i as f32 * (bh + gap)),
                egui::Vec2::new(bw, bh),
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

        Self::text(
            ui,
            ep::Pos2::new(cx, by0 + labels.len() as f32 * (bh + gap) + 12.0),
            "Rust + wgpu · 程序化地形 · 实时方块渲染",
            11.0,
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 105),
        );
        Self::draw_footer(ui, screen, "mc 0.1.0 · rust + wgpu", "seed 随机生成");
    }

    fn draw_worlds(&mut self, ui: &mut egui::Ui, screen: ep::Rect) {
        let w = screen.width();
        let h = screen.height();
        let cx = screen.center().x;
        let pw = w.clamp(360.0, 760.0);
        let ph = (h * 0.78).clamp(430.0, 620.0);
        let px = cx - pw / 2.0;
        let py = (h - ph) / 2.0;
        let panel = ep::Rect::from_min_size(ep::Pos2::new(px, py), egui::Vec2::new(pw, ph));
        Self::panel(ui, panel);
        Self::text(
            ui,
            ep::Pos2::new(cx, py + 36.0),
            "选择世界",
            30.0,
            egui::Color32::WHITE,
        );
        Self::text(
            ui,
            ep::Pos2::new(cx, py + 66.0),
            "继续一段已经生成的世界，或开启新的种子",
            13.0,
            egui::Color32::from_rgb(0xB4, 0xB4, 0xB4),
        );

        let row_x = px + 24.0;
        let row_w = pw - 48.0;
        let list_top = py + 96.0;
        let list_bottom = py + ph - 92.0;
        let row_h = 64.0;
        let row_gap = 10.0;
        let max_rows = (((list_bottom - list_top) / (row_h + row_gap)).floor() as usize).max(1);
        let mut enter_index = None;

        if self.worlds.is_empty() {
            let icon_center = ep::Pos2::new(cx, py + ph * 0.42);
            Self::draw_voxel_cube(ui, icon_center, 37.0, 28.0, self.time);
            Self::text(
                ui,
                ep::Pos2::new(cx, py + ph * 0.42 + 80.0),
                "还没有世界",
                20.0,
                egui::Color32::from_rgb(0xD0, 0xD0, 0xD0),
            );
            Self::text(
                ui,
                ep::Pos2::new(cx, py + ph * 0.42 + 109.0),
                "创建一个世界开始游戏",
                13.0,
                egui::Color32::from_rgb(0x9A, 0x9A, 0x9A),
            );
        } else {
            for (i, world) in self.worlds.iter().enumerate().take(max_rows) {
                let row_y = list_top + i as f32 * (row_h + row_gap);
                let row = ep::Rect::from_min_size(
                    ep::Pos2::new(row_x, row_y),
                    egui::Vec2::new(row_w, row_h),
                );
                let response = ui.allocate_rect(row, egui::Sense::click());
                let hover = ui
                    .ctx()
                    .animate_bool_with_time(response.id, response.hovered(), 0.12);
                let painter = ui.painter();
                let bg = Self::color_lerp(
                    egui::Color32::from_rgba_unmultiplied(32, 24, 16, 218),
                    egui::Color32::from_rgba_unmultiplied(56, 42, 27, 235),
                    hover,
                );
                let border = Self::color_lerp(
                    egui::Color32::from_rgb(0x55, 0x3C, 0x24),
                    egui::Color32::from_rgb(0xB2, 0x92, 0x5A),
                    hover,
                );
                painter.rect_filled(row, 3.0, bg);
                painter.rect_stroke(
                    row,
                    3.0,
                    egui::Stroke::new(1.5, border),
                    egui::StrokeKind::Inside,
                );

                let icon = ep::Rect::from_min_size(
                    ep::Pos2::new(row.min.x + 12.0, row.min.y + 9.0),
                    egui::Vec2::splat(46.0),
                );
                Self::draw_world_icon(ui, icon, &self.atlas, self.atlas_texture.id(), world.seed);

                Self::left_text(
                    ui,
                    ep::Pos2::new(icon.max.x + 14.0, row.min.y + 19.0),
                    world.name.clone(),
                    17.0,
                    egui::Color32::WHITE,
                );
                Self::left_text(
                    ui,
                    ep::Pos2::new(icon.max.x + 14.0, row.min.y + 43.0),
                    format!(
                        "种子 {} · 出生点 {}, {}, {}",
                        world.seed, world.spawn_x, world.spawn_height, world.spawn_z
                    ),
                    11.5,
                    egui::Color32::from_rgb(0xA6, 0xA6, 0xA6),
                );

                let enter = ep::Rect::from_min_size(
                    ep::Pos2::new(row.max.x - 94.0, row.min.y + 15.0),
                    egui::Vec2::new(80.0, 34.0),
                );
                if Self::menu_button(ui, enter, "进入", true) || response.clicked() {
                    enter_index = Some(i);
                }
            }

            if self.worlds.len() > max_rows {
                Self::text(
                    ui,
                    ep::Pos2::new(cx, list_bottom + 6.0),
                    format!("还有 {} 个世界未显示", self.worlds.len() - max_rows),
                    11.0,
                    egui::Color32::from_rgb(0x8F, 0x8F, 0x8F),
                );
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
            egui::Vec2::new(button_w, 36.0),
        );
        let back = ep::Rect::from_min_size(
            ep::Pos2::new(px + 40.0 + button_w, bottom_y),
            egui::Vec2::new(button_w, 36.0),
        );
        if Self::menu_button(ui, create, "创建新世界", true) {
            self.form_error = None;
            self.draft_seed = random_world_seed().to_string();
            self.screen = LauncherScreen::CreateWorld;
        }
        if Self::menu_button(ui, back, "返回", true) {
            self.screen = LauncherScreen::Home;
        }
    }

    fn draw_world_icon(
        ui: &egui::Ui,
        rect: ep::Rect,
        atlas: &Atlas,
        texture_id: egui::TextureId,
        seed: u64,
    ) {
        let painter = ui.painter();
        painter.rect_filled(
            ep::Rect::from_min_size(rect.min + egui::Vec2::new(2.0, 3.0), rect.size()),
            3.0,
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, 105),
        );
        painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(0x18, 0x14, 0x10));

        let tile = atlas
            .tile_index("grass_block_side_overlay.png")
            .or_else(|| atlas.tile_index("grass_block_top.png"))
            .unwrap_or(0);
        let (u0, v0, u1, v1) = atlas.uv(tile);
        let uv = ep::Rect::from_min_max(ep::Pos2::new(u0, v0), ep::Pos2::new(u1, v1));
        painter.image(texture_id, rect.shrink(1.5), uv, egui::Color32::WHITE);
        painter.rect_stroke(
            rect,
            3.0,
            egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 38),
            ),
            egui::StrokeKind::Inside,
        );

        for i in 0..4 {
            let x = Self::hash01(seed ^ (i as u64 * 0x9E37_79B9)) * rect.width();
            let y = Self::hash01(seed ^ (i as u64 * 0x85EB_CA6B)) * rect.height();
            let highlight = egui::Color32::from_rgba_unmultiplied(
                255,
                255,
                255,
                18 + (Self::hash01(seed ^ i as u64) * 28.0) as u8,
            );
            painter.rect_filled(
                ep::Rect::from_min_size(
                    ep::Pos2::new(rect.min.x + x, rect.min.y + y),
                    egui::Vec2::splat(1.5),
                ),
                0.0,
                highlight,
            );
        }
    }

    fn draw_create_world(&mut self, ui: &mut egui::Ui, screen: ep::Rect) {
        let w = screen.width();
        let h = screen.height();
        let cx = screen.center().x;
        let pw = w.clamp(360.0, 560.0);
        let ph = (h - 40.0).clamp(430.0, 540.0);
        let px = cx - pw / 2.0;
        let py = (h - ph) / 2.0;
        Self::panel(
            ui,
            ep::Rect::from_min_size(ep::Pos2::new(px, py), egui::Vec2::new(pw, ph)),
        );
        Self::text(
            ui,
            ep::Pos2::new(cx, py + 36.0),
            "创建新世界",
            29.0,
            egui::Color32::WHITE,
        );
        Self::text(
            ui,
            ep::Pos2::new(cx, py + 66.0),
            "相同的种子会生成相同的地形",
            12.5,
            egui::Color32::from_rgb(0xA8, 0xA8, 0xA8),
        );

        let label_x = px + 32.0;
        let field_x = px + 32.0;
        let field_w = pw - 64.0;
        Self::left_text(
            ui,
            ep::Pos2::new(label_x, py + 104.0),
            "世界名称",
            14.0,
            egui::Color32::from_rgb(0xD8, 0xD8, 0xD8),
        );
        let name_rect = ep::Rect::from_min_size(
            ep::Pos2::new(field_x, py + 120.0),
            egui::Vec2::new(field_w, 38.0),
        );
        ui.put(
            name_rect,
            egui::TextEdit::singleline(&mut self.draft_name).hint_text("世界名称"),
        );

        Self::left_text(
            ui,
            ep::Pos2::new(label_x, py + 183.0),
            "世界种子",
            14.0,
            egui::Color32::from_rgb(0xD8, 0xD8, 0xD8),
        );
        let seed_w = field_w - 78.0;
        let seed_rect = ep::Rect::from_min_size(
            ep::Pos2::new(field_x, py + 199.0),
            egui::Vec2::new(seed_w, 38.0),
        );
        ui.put(
            seed_rect,
            egui::TextEdit::singleline(&mut self.draft_seed).hint_text("请输入数字种子"),
        );
        let random_rect = ep::Rect::from_min_size(
            ep::Pos2::new(field_x + seed_w + 10.0, py + 199.0),
            egui::Vec2::new(68.0, 38.0),
        );
        if Self::menu_button(ui, random_rect, "随机", true) {
            self.draft_seed = random_world_seed().to_string();
        }

        Self::left_text(
            ui,
            ep::Pos2::new(label_x, py + 261.0),
            "种子支持非负整数；留空则随机生成。",
            12.0,
            egui::Color32::from_rgb(0x9A, 0x9A, 0x9A),
        );

        if let Some(error) = self.form_error.as_deref() {
            Self::left_text(
                ui,
                ep::Pos2::new(label_x, py + 296.0),
                error,
                13.0,
                egui::Color32::from_rgb(0xFF, 0xA0, 0x80),
            );
        }

        let button_w = (pw - 64.0) / 2.0;
        let bottom_y = py + ph - 58.0;
        let create = ep::Rect::from_min_size(
            ep::Pos2::new(px + 24.0, bottom_y),
            egui::Vec2::new(button_w, 36.0),
        );
        let back = ep::Rect::from_min_size(
            ep::Pos2::new(px + 40.0 + button_w, bottom_y),
            egui::Vec2::new(button_w, 36.0),
        );
        if Self::menu_button(ui, create, "创建并进入", true) {
            let name = self.draft_name.trim().to_string();
            let seed_text = self.draft_seed.trim();
            if name.is_empty() {
                self.form_error = Some("世界名称不能为空".to_string());
            } else if seed_text.is_empty() {
                self.start_world_creation(name, random_world_seed());
            } else if let Ok(seed) = seed_text.parse::<u64>() {
                self.start_world_creation(name, seed);
            } else {
                self.form_error = Some("世界种子必须是非负整数".to_string());
            }
        }
        if Self::menu_button(ui, back, "返回", true) {
            self.form_error = None;
            self.screen = LauncherScreen::Worlds;
        }
    }

    fn draw_creating_world(&self, ui: &mut egui::Ui, screen: ep::Rect) {
        let painter = ui.painter();
        painter.rect_filled(
            screen,
            0.0,
            egui::Color32::from_rgba_unmultiplied(4, 6, 9, 82),
        );

        let center = screen.center();
        let card_w = (screen.width() - 48.0).clamp(320.0, 560.0);
        let card = ep::Rect::from_center_size(center, egui::Vec2::new(card_w, 210.0));
        Self::panel(ui, card);

        let pulse = ((self.time * 2.2).sin() * 0.5 + 0.5) as f32;
        Self::text(
            ui,
            ep::Pos2::new(center.x, card.min.y + 42.0),
            "正在生成世界",
            29.0,
            egui::Color32::WHITE,
        );
        Self::text(
            ui,
            ep::Pos2::new(center.x, card.min.y + 76.0),
            self.creation_progress.stage,
            14.0,
            egui::Color32::from_rgb(0xC0, 0xC0, 0xC0),
        );

        let bar = ep::Rect::from_center_size(
            ep::Pos2::new(center.x, card.min.y + 118.0),
            egui::Vec2::new(card_w - 72.0, 24.0),
        );
        painter.rect_filled(bar, 2.0, egui::Color32::from_rgb(0x08, 0x08, 0x08));
        painter.rect_stroke(
            bar,
            2.0,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(0x6A, 0x6A, 0x6A)),
            egui::StrokeKind::Inside,
        );

        let fill_width = (bar.width() - 6.0) * self.creation_display_progress;
        if fill_width > 0.0 {
            let fill = ep::Rect::from_min_size(
                bar.min + egui::Vec2::splat(3.0),
                egui::Vec2::new(fill_width, bar.height() - 6.0),
            );
            Self::vertical_gradient(
                painter,
                fill,
                egui::Color32::from_rgb(0x7B, 0xC7, 0x4B),
                egui::Color32::from_rgb(0x3F, 0x84, 0x2C),
            );

            let shimmer_x = fill.left() + fill.width() * pulse;
            let shimmer = ep::Rect::from_min_max(
                ep::Pos2::new((shimmer_x - 18.0).max(fill.left()), fill.top()),
                ep::Pos2::new((shimmer_x + 18.0).min(fill.right()), fill.bottom()),
            );
            painter.rect_filled(
                shimmer,
                0.0,
                egui::Color32::from_rgba_unmultiplied(225, 255, 190, 45),
            );
        }

        let percent = self.creation_display_progress * 100.0;
        Self::text(
            ui,
            ep::Pos2::new(center.x, card.min.y + 151.0),
            format!("{percent:02.0}%"),
            13.0,
            egui::Color32::from_rgb(0xAB, 0xAB, 0xAB),
        );

        let dot_count = (self.time * 2.0).floor() as usize % 4;
        Self::text(
            ui,
            ep::Pos2::new(center.x, card.min.y + 181.0),
            format!("请稍候{}", ".".repeat(dot_count)),
            13.0,
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, (135.0 + pulse * 85.0) as u8),
        );

        Self::draw_footer(ui, screen, "正在构建世界 · 请勿关闭窗口", "Rust + wgpu");
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
            "WASD 移动 · 鼠标视角 · 左键挖掘 · 右键放置 · Space 跳跃/上浮 · Shift 下潜 · F4 切换模式 · E 物品栏 · J JEI · Esc 返回",
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
                    if self.game_mode.is_creative() {
                        self.game_mode.display()
                    } else if player.on_ground {
                        "地面"
                    } else {
                        "空中"
                    },
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
            self.draw_inventory_modern(ui, screen);
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
            self.draw_inventory_slot(ui, INVENTORY_SIZE - HOTBAR_SIZE + index, rect, true);
        }
        Self::text(
            ui,
            ep::Pos2::new(screen.center().x, y - 12.0),
            "1–9 选择 · E 打开物品栏",
            11.0,
            egui::Color32::from_rgba_premultiplied(255, 255, 255, 165),
        );
    }

    fn draw_inventory_modern(&mut self, ui: &mut egui::Ui, screen: ep::Rect) {
        const WIDTH: f32 = 728.0;
        const HEIGHT: f32 = 692.0;
        const SLOT: f32 = 72.0;
        const GAP: f32 = 4.0;
        let scale = ((screen.width() - 18.0) / WIDTH)
            .min((screen.height() - 18.0) / HEIGHT)
            .clamp(0.5, 1.0);
        let inventory_screen = if self.jei_visible {
            ep::Rect::from_min_max(
                screen.min,
                ep::Pos2::new(screen.max.x - JEI_PANEL_WIDTH - 18.0, screen.max.y),
            )
        } else {
            screen
        };
        let origin = inventory_screen.center() - ep::Vec2::new(WIDTH, HEIGHT) * scale / 2.0;
        let local = |x: f32, y: f32, width: f32, height: f32| {
            ep::Rect::from_min_size(
                origin + ep::Vec2::new(x, y) * scale,
                ep::Vec2::new(width, height) * scale,
            )
        };

        ui.painter().rect_filled(
            screen,
            0.0,
            egui::Color32::from_rgba_premultiplied(0, 0, 0, 150),
        );
        Self::draw_vanilla_panel(ui, local(0.0, 0.0, WIDTH, HEIGHT));
        for row in 0..4 {
            Self::draw_vanilla_slot(ui, local(24.0, 22.0 + row as f32 * 76.0, SLOT, SLOT), false);
        }
        Self::draw_character_preview(ui, local(101.0, 20.0, 224.0, 304.0));
        Self::draw_vanilla_slot(ui, local(325.0, 250.0, SLOT, SLOT), false);

        for row in 0..2 {
            for column in 0..2 {
                Self::draw_vanilla_slot(
                    ui,
                    local(
                        401.0 + column as f32 * 76.0,
                        58.0 + row as f32 * 76.0,
                        SLOT,
                        SLOT,
                    ),
                    false,
                );
            }
        }
        let arrow = local(563.0, 104.0, 54.0, 42.0);
        let arrow_y = arrow.center().y;
        ui.painter().add(ep::Shape::convex_polygon(
            vec![
                ep::Pos2::new(arrow.min.x, arrow_y - 7.0 * scale),
                ep::Pos2::new(arrow.max.x - 17.0 * scale, arrow_y - 7.0 * scale),
                ep::Pos2::new(arrow.max.x - 17.0 * scale, arrow_y - 15.0 * scale),
                ep::Pos2::new(arrow.max.x, arrow_y),
                ep::Pos2::new(arrow.max.x - 17.0 * scale, arrow_y + 15.0 * scale),
                ep::Pos2::new(arrow.max.x - 17.0 * scale, arrow_y + 7.0 * scale),
                ep::Pos2::new(arrow.min.x, arrow_y + 7.0 * scale),
            ],
            egui::Color32::from_rgb(0x8B, 0x8B, 0x8B),
            egui::Stroke::NONE,
        ));
        Self::draw_vanilla_slot(ui, local(633.0, 96.0, SLOT, SLOT), false);

        let mut hovered_slot = false;
        for row in 0..3 {
            for column in 0..HOTBAR_SIZE {
                let index = row * HOTBAR_SIZE + column;
                let rect = local(
                    24.0 + column as f32 * (SLOT + GAP),
                    344.0 + row as f32 * (SLOT + GAP),
                    SLOT,
                    SLOT,
                );
                hovered_slot |= self.draw_drag_slot(ui, index, rect);
            }
        }
        for column in 0..HOTBAR_SIZE {
            let rect = local(24.0 + column as f32 * (SLOT + GAP), 590.0, SLOT, SLOT);
            hovered_slot |= self.draw_drag_slot(ui, INVENTORY_SIZE - HOTBAR_SIZE + column, rect);
        }
        let pointer_released = ui.input(|input| {
            input.pointer.button_released(egui::PointerButton::Primary)
                || input
                    .pointer
                    .button_released(egui::PointerButton::Secondary)
        });
        if pointer_released && self.cursor_stack.is_some() && !hovered_slot {
            self.return_cursor_stack();
        }
        if let Some(stack) = self.cursor_stack
            && let Some(pointer) = ui.ctx().pointer_interact_pos()
        {
            self.draw_stack(
                ui,
                stack,
                ep::Rect::from_center_size(pointer, ep::Vec2::splat(66.0 * scale)),
            );
        }
        if self.jei_visible {
            self.draw_jei_panel(ui, screen);
        }
    }

    fn draw_vanilla_panel(ui: &egui::Ui, rect: ep::Rect) {
        let painter = ui.painter();
        painter.rect_filled(rect, 13.0, egui::Color32::from_rgb(0x12, 0x12, 0x12));
        painter.rect_stroke(
            rect,
            13.0,
            egui::Stroke::new(2.0, egui::Color32::BLACK),
            egui::StrokeKind::Inside,
        );
        let inner = rect.shrink(7.0);
        painter.rect_filled(inner, 8.0, egui::Color32::from_rgb(0xC6, 0xC6, 0xC6));
        painter.line_segment(
            [inner.left_top(), ep::Pos2::new(inner.right(), inner.top())],
            egui::Stroke::new(3.0, egui::Color32::from_rgb(0xF5, 0xF5, 0xF5)),
        );
        painter.line_segment(
            [
                inner.left_top(),
                ep::Pos2::new(inner.left(), inner.bottom()),
            ],
            egui::Stroke::new(3.0, egui::Color32::from_rgb(0xF5, 0xF5, 0xF5)),
        );
        painter.line_segment(
            [
                inner.right_bottom(),
                ep::Pos2::new(inner.left(), inner.bottom()),
            ],
            egui::Stroke::new(3.0, egui::Color32::from_rgb(0x6B, 0x6B, 0x6B)),
        );
        painter.line_segment(
            [
                inner.right_bottom(),
                ep::Pos2::new(inner.right(), inner.top()),
            ],
            egui::Stroke::new(3.0, egui::Color32::from_rgb(0x6B, 0x6B, 0x6B)),
        );
    }

    fn draw_vanilla_slot(ui: &egui::Ui, rect: ep::Rect, selected: bool) {
        let painter = ui.painter();
        painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(0x38, 0x38, 0x38));
        let inner = rect.shrink(4.0);
        painter.rect_filled(inner, 0.0, egui::Color32::from_rgb(0x8B, 0x8B, 0x8B));
        painter.line_segment(
            [inner.left_top(), ep::Pos2::new(inner.right(), inner.top())],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(0xE5, 0xE5, 0xE5)),
        );
        painter.line_segment(
            [
                inner.left_top(),
                ep::Pos2::new(inner.left(), inner.bottom()),
            ],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(0xE5, 0xE5, 0xE5)),
        );
        painter.line_segment(
            [
                inner.right_bottom(),
                ep::Pos2::new(inner.left(), inner.bottom()),
            ],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(0x4A, 0x4A, 0x4A)),
        );
        painter.line_segment(
            [
                inner.right_bottom(),
                ep::Pos2::new(inner.right(), inner.top()),
            ],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(0x4A, 0x4A, 0x4A)),
        );
        if selected {
            painter.rect_stroke(
                rect.shrink(1.0),
                0.0,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(0xFF, 0xF2, 0x88)),
                egui::StrokeKind::Inside,
            );
        }
    }

    fn draw_character_preview(ui: &egui::Ui, rect: ep::Rect) {
        let painter = ui.painter();
        let scale = (rect.width() / 224.0).min(rect.height() / 304.0);
        let center = rect.center().x;
        let px = |x: f32, y: f32, width: f32, height: f32| {
            ep::Rect::from_min_size(
                ep::Pos2::new(center + x * scale, rect.min.y + y * scale),
                ep::Vec2::new(width, height) * scale,
            )
        };
        // 像素化角色预览，保持与方块图标一致的低分辨率视觉。
        painter.rect_filled(
            px(-31.0, 18.0, 62.0, 62.0),
            1.0,
            egui::Color32::from_rgb(0x25, 0x22, 0x23),
        );
        painter.rect_filled(
            px(-27.0, 78.0, 54.0, 47.0),
            0.0,
            egui::Color32::from_rgb(0x79, 0x5A, 0x14),
        );
        painter.rect_filled(
            px(-41.0, 84.0, 14.0, 45.0),
            0.0,
            egui::Color32::from_rgb(0x8A, 0x6A, 0x20),
        );
        painter.rect_filled(
            px(27.0, 84.0, 14.0, 45.0),
            0.0,
            egui::Color32::from_rgb(0x8A, 0x6A, 0x20),
        );
        painter.rect_filled(
            px(-39.0, 119.0, 12.0, 28.0),
            0.0,
            egui::Color32::from_rgb(0xF4, 0xDE, 0xC1),
        );
        painter.rect_filled(
            px(27.0, 119.0, 12.0, 28.0),
            0.0,
            egui::Color32::from_rgb(0xF4, 0xDE, 0xC1),
        );
        painter.rect_filled(
            px(-25.0, 125.0, 24.0, 61.0),
            0.0,
            egui::Color32::from_rgb(0x29, 0x2A, 0x2D),
        );
        painter.rect_filled(
            px(1.0, 125.0, 24.0, 61.0),
            0.0,
            egui::Color32::from_rgb(0x29, 0x2A, 0x2D),
        );
        painter.rect_filled(
            px(-26.0, 179.0, 25.0, 18.0),
            0.0,
            egui::Color32::from_rgb(0x5F, 0x47, 0x1A),
        );
        painter.rect_filled(
            px(1.0, 179.0, 25.0, 18.0),
            0.0,
            egui::Color32::from_rgb(0x5F, 0x47, 0x1A),
        );
        painter.rect_filled(
            px(-25.0, 86.0, 50.0, 36.0),
            0.0,
            egui::Color32::from_rgb(0x86, 0x66, 0x15),
        );
        painter.rect_filled(
            px(-27.0, 47.0, 54.0, 38.0),
            0.0,
            egui::Color32::from_rgb(0xF4, 0xDE, 0xC1),
        );
        painter.rect_filled(
            px(-31.0, 20.0, 62.0, 31.0),
            0.0,
            egui::Color32::from_rgb(0x25, 0x22, 0x23),
        );
        painter.rect_filled(
            px(-31.0, 35.0, 9.0, 20.0),
            0.0,
            egui::Color32::from_rgb(0x25, 0x22, 0x23),
        );
        painter.rect_filled(
            px(22.0, 35.0, 9.0, 20.0),
            0.0,
            egui::Color32::from_rgb(0x25, 0x22, 0x23),
        );
        painter.rect_filled(
            px(-13.0, 54.0, 8.0, 8.0),
            0.0,
            egui::Color32::from_rgb(0x2B, 0x27, 0x25),
        );
        painter.rect_filled(
            px(9.0, 54.0, 8.0, 8.0),
            0.0,
            egui::Color32::from_rgb(0x2B, 0x27, 0x25),
        );
    }

    fn add_textured_face(
        mesh: &mut ep::Mesh,
        points: [ep::Pos2; 4],
        uv: (f32, f32, f32, f32),
        color: egui::Color32,
    ) {
        let (u0, v0, u1, v1) = uv;
        let start = mesh.vertices.len() as u32;
        for (pos, tex) in points.into_iter().zip([
            ep::Pos2::new(u0, v0),
            ep::Pos2::new(u1, v0),
            ep::Pos2::new(u1, v1),
            ep::Pos2::new(u0, v1),
        ]) {
            mesh.vertices.push(ep::Vertex {
                pos,
                uv: tex,
                color,
            });
        }
        mesh.add_triangle(start, start + 1, start + 2);
        mesh.add_triangle(start, start + 2, start + 3);
    }

    fn draw_stack(&self, ui: &egui::Ui, stack: ItemStack, rect: ep::Rect) {
        let size = rect.width().min(rect.height()) * 0.86;
        let cx = rect.center().x;
        let cy = rect.center().y - size * 0.02;
        let top = [
            ep::Pos2::new(cx, cy - size * 0.30),
            ep::Pos2::new(cx + size * 0.31, cy - size * 0.14),
            ep::Pos2::new(cx, cy + size * 0.02),
            ep::Pos2::new(cx - size * 0.31, cy - size * 0.14),
        ];
        let left = [
            top[3],
            top[2],
            ep::Pos2::new(cx, cy + size * 0.34),
            ep::Pos2::new(cx - size * 0.31, cy + size * 0.18),
        ];
        let right = [
            top[2],
            top[1],
            ep::Pos2::new(cx + size * 0.31, cy + size * 0.18),
            ep::Pos2::new(cx, cy + size * 0.34),
        ];
        let top_uv = self
            .atlas
            .uv(self.atlas.tile_id(stack.block, stack.block.tile(Face::Top)));
        let side_uv = self.atlas.uv(self
            .atlas
            .tile_id(stack.block, stack.block.tile(Face::Side)));
        let mut mesh = ep::Mesh::with_texture(self.atlas_texture.id());
        Self::add_textured_face(
            &mut mesh,
            left,
            side_uv,
            egui::Color32::from_rgb(0xD0, 0xD0, 0xD0),
        );
        Self::add_textured_face(
            &mut mesh,
            right,
            side_uv,
            egui::Color32::from_rgb(0xA0, 0xA0, 0xA0),
        );
        Self::add_textured_face(&mut mesh, top, top_uv, egui::Color32::WHITE);
        ui.painter().add(ep::Shape::mesh(mesh));
        let pos = rect.right_bottom() - ep::Vec2::new(5.0, 4.0);
        let font = egui::FontId::proportional((16.0 * (rect.width() / 72.0)).max(10.0));
        ui.painter().text(
            pos + ep::Vec2::new(1.0, 1.0),
            egui::Align2::RIGHT_BOTTOM,
            stack.count.to_string(),
            font.clone(),
            egui::Color32::BLACK,
        );
        ui.painter().text(
            pos,
            egui::Align2::RIGHT_BOTTOM,
            stack.count.to_string(),
            font,
            egui::Color32::WHITE,
        );
    }

    fn draw_drag_slot(&mut self, ui: &mut egui::Ui, index: usize, rect: ep::Rect) -> bool {
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let hovered = response.hovered();
        Self::draw_vanilla_slot(ui, rect, self.inventory.selected == index);
        if let Some(stack) = self.inventory.slots[index] {
            if self.cursor_source != Some(index) {
                self.draw_stack(ui, stack, rect);
            }
            if hovered {
                response.clone().on_hover_text(stack.block.def().display);
            }
        }
        if response.drag_started_by(egui::PointerButton::Primary) && self.cursor_stack.is_none() {
            self.cursor_stack = self.inventory.take_stack(index);
            self.cursor_source = Some(index);
        } else if response.clicked_by(egui::PointerButton::Primary) {
            if self.cursor_stack.is_none() {
                self.cursor_stack = self.inventory.take_stack(index);
                self.cursor_source = Some(index);
            } else {
                self.inventory.put_stack(index, &mut self.cursor_stack);
                self.cursor_source = None;
            }
        } else if response.clicked_by(egui::PointerButton::Secondary) {
            if self.cursor_stack.is_none() {
                self.cursor_stack = self.inventory.take_half(index);
                self.cursor_source = Some(index);
            } else {
                self.inventory.put_one(index, &mut self.cursor_stack);
                self.cursor_source = None;
            }
        } else if self.cursor_stack.is_some()
            && ui.input(|input| input.pointer.button_released(egui::PointerButton::Primary))
            && ui
                .ctx()
                .pointer_interact_pos()
                .is_some_and(|pointer| rect.contains(pointer))
        {
            self.inventory.put_stack(index, &mut self.cursor_stack);
            self.cursor_source = None;
        }
        hovered
    }

    #[allow(dead_code)]
    fn draw_inventory_legacy(&mut self, ui: &mut egui::Ui, screen: ep::Rect) {
        let inventory_screen = if self.jei_visible {
            ep::Rect::from_min_max(
                screen.min,
                ep::Pos2::new(screen.max.x - JEI_PANEL_WIDTH - 18.0, screen.max.y),
            )
        } else {
            screen
        };
        const SLOT: f32 = 42.0;
        const GAP: f32 = 4.0;
        const COLUMNS: usize = HOTBAR_SIZE;
        let grid_width = COLUMNS as f32 * SLOT + (COLUMNS - 1) as f32 * GAP;
        let panel_width = grid_width + 42.0;
        let panel_height: f32 = 274.0;
        let panel = ep::Rect::from_center_size(
            inventory_screen.center(),
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
                self.draw_inventory_slot(ui, index, rect, true);
            }
        }

        let hotbar_y = panel.max.y - SLOT - 20.0;
        for column in 0..HOTBAR_SIZE {
            let rect = ep::Rect::from_min_size(
                ep::Pos2::new(x + column as f32 * (SLOT + GAP), hotbar_y),
                egui::Vec2::splat(SLOT),
            );
            self.draw_inventory_slot(ui, INVENTORY_SIZE - HOTBAR_SIZE + column, rect, true);
        }
        Self::left_text(
            ui,
            ep::Pos2::new(panel.min.x + 20.0, panel.max.y - 5.0),
            "点击格子选择方块",
            11.0,
            egui::Color32::from_rgb(0xB0, 0xB0, 0xB0),
        );
        if self.jei_visible {
            self.draw_jei_panel(ui, screen);
        }
    }

    fn jei_matches(&self) -> Vec<Block> {
        let query = self.jei_search.trim().to_lowercase();
        crate::world::block::ALL
            .into_iter()
            .filter(|block| *block != Block::Air)
            .filter(|block| {
                query.is_empty()
                    || block.def().display.to_lowercase().contains(&query)
                    || block.def().name.to_lowercase().contains(&query)
            })
            .collect()
    }

    fn draw_recipe_detail(&mut self, ui: &mut egui::Ui, panel: ep::Rect, block: Block) {
        Self::left_text(
            ui,
            ep::Pos2::new(panel.min.x + 12.0, panel.min.y + 60.0),
            format!("{} · {}", block.def().display, block.def().name),
            13.0,
            egui::Color32::WHITE,
        );

        let recipes: Vec<&'static Recipe> = recipe::for_output(block).collect();
        let usages: Vec<&'static Recipe> = recipe::using(block).collect();
        let mut y = panel.min.y + 78.0;
        if recipes.is_empty() {
            Self::left_text(
                ui,
                ep::Pos2::new(panel.min.x + 12.0, y),
                "没有已知的制作配方",
                12.0,
                egui::Color32::from_rgb(0xA8, 0xA8, 0xA8),
            );
            y += 27.0;
        } else {
            Self::left_text(
                ui,
                ep::Pos2::new(panel.min.x + 12.0, y),
                "获得方式",
                12.0,
                egui::Color32::from_rgb(0xD8, 0xD8, 0xD8),
            );
            y += 10.0;
            self.draw_recipe_row(ui, panel, y, recipes[0]);
            y += 111.0;
        }

        if usages.is_empty() {
            Self::left_text(
                ui,
                ep::Pos2::new(panel.min.x + 12.0, y),
                "没有已知用途",
                12.0,
                egui::Color32::from_rgb(0xA8, 0xA8, 0xA8),
            );
        } else {
            Self::left_text(
                ui,
                ep::Pos2::new(panel.min.x + 12.0, y),
                "用途",
                12.0,
                egui::Color32::from_rgb(0xD8, 0xD8, 0xD8),
            );
            y += 10.0;
            for usage in usages.into_iter().take(2) {
                self.draw_recipe_row(ui, panel, y, usage);
                y += 111.0;
            }
        }
    }

    fn draw_recipe_row(&self, ui: &mut egui::Ui, panel: ep::Rect, y: f32, recipe: &Recipe) {
        let card = ep::Rect::from_min_size(
            ep::Pos2::new(panel.min.x + 7.0, y),
            ep::Vec2::new(panel.width() - 14.0, 101.0),
        );
        ui.painter()
            .rect_filled(card, 2.0, egui::Color32::from_rgb(0x2B, 0x2D, 0x2F));
        ui.painter().rect_stroke(
            card,
            2.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0x4D, 0x50, 0x53)),
            egui::StrokeKind::Inside,
        );

        let slot = 20.0;
        let gap = 2.0;
        let grid_width = slot * 3.0 + gap * 2.0;
        let grid_x = card.min.x + 12.0;
        let grid_y = card.min.y + 17.0;
        if recipe.kind == RecipeKind::Crafting {
            for index in 0..recipe::CRAFTING_SLOTS {
                let row = index / 3;
                let column = index % 3;
                let rect = ep::Rect::from_min_size(
                    ep::Pos2::new(
                        grid_x + column as f32 * (slot + gap),
                        grid_y + row as f32 * (slot + gap),
                    ),
                    ep::Vec2::splat(slot),
                );
                self.draw_recipe_slot(ui, rect, recipe.ingredients[index]);
            }
        } else {
            self.draw_recipe_slot(
                ui,
                ep::Rect::from_min_size(
                    ep::Pos2::new(grid_x + grid_width / 2.0 - slot / 2.0, grid_y + 21.0),
                    ep::Vec2::splat(slot),
                ),
                recipe.ingredients[0],
            );
        }

        let arrow_start = card.min.x + 91.0;
        let arrow_y = card.center().y - 2.0;
        ui.painter().line_segment(
            [
                ep::Pos2::new(arrow_start, arrow_y),
                ep::Pos2::new(arrow_start + 25.0, arrow_y),
            ],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(0xB8, 0xB8, 0xB8)),
        );
        ui.painter().add(ep::Shape::convex_polygon(
            vec![
                ep::Pos2::new(arrow_start + 25.0, arrow_y - 6.0),
                ep::Pos2::new(arrow_start + 34.0, arrow_y),
                ep::Pos2::new(arrow_start + 25.0, arrow_y + 6.0),
            ],
            egui::Color32::from_rgb(0xB8, 0xB8, 0xB8),
            egui::Stroke::NONE,
        ));
        let output_rect = ep::Rect::from_min_size(
            ep::Pos2::new(card.max.x - 42.0, card.min.y + 25.0),
            egui::Vec2::splat(32.0),
        );
        self.draw_recipe_slot(ui, output_rect, Some(recipe.output));
        Self::text(
            ui,
            ep::Pos2::new(output_rect.center().x, card.max.y - 11.0),
            format!("×{}", recipe.output_count),
            11.0,
            egui::Color32::from_rgb(0xE0, 0xE0, 0xE0),
        );
        Self::left_text(
            ui,
            ep::Pos2::new(card.min.x + 8.0, card.max.y - 11.0),
            recipe.kind.display(),
            10.0,
            egui::Color32::from_rgb(0xA8, 0xA8, 0xA8),
        );
    }

    fn draw_recipe_slot(&self, ui: &mut egui::Ui, rect: ep::Rect, block: Option<Block>) {
        Self::draw_vanilla_slot(ui, rect, false);
        if let Some(block) = block {
            self.draw_jei_icon(ui, rect.shrink(2.0), block, Face::Side);
        }
    }

    fn draw_jei_panel(&mut self, ui: &mut egui::Ui, screen: ep::Rect) {
        let height = (screen.height() - 14.0).min(JEI_PANEL_HEIGHT);
        let panel = ep::Rect::from_center_size(
            ep::Pos2::new(
                screen.max.x - JEI_PANEL_WIDTH / 2.0 - 7.0,
                screen.center().y,
            ),
            ep::Vec2::new(JEI_PANEL_WIDTH, height),
        );
        ui.painter().rect_filled(
            panel,
            1.0,
            egui::Color32::from_rgba_premultiplied(20, 22, 24, 238),
        );
        ui.painter().rect_stroke(
            panel,
            1.0,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(0x55, 0x58, 0x5C)),
            egui::StrokeKind::Inside,
        );
        ui.painter().rect_stroke(
            panel.shrink(4.0),
            0.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0x0C, 0x0D, 0x0E)),
            egui::StrokeKind::Inside,
        );

        let button_size = 38.0;
        let button_y = panel.min.y + 6.0;
        let previous = ep::Rect::from_min_size(
            ep::Pos2::new(panel.min.x + 5.0, button_y),
            ep::Vec2::splat(button_size),
        );
        let next = ep::Rect::from_min_size(
            ep::Pos2::new(panel.max.x - button_size - 5.0, button_y),
            ep::Vec2::splat(button_size),
        );
        let matches = self.jei_matches();
        let slot = 29.0;
        let gap = 3.0;
        let grid_width = JEI_COLUMNS as f32 * slot + (JEI_COLUMNS - 1) as f32 * gap;
        let grid_x = panel.center().x - grid_width / 2.0;
        let grid_top = panel.min.y + 51.0;
        let grid_bottom = panel.max.y - 48.0;
        let rows = (((grid_bottom - grid_top + gap) / (slot + gap)).floor() as usize).max(1);
        let visible_items = rows * JEI_COLUMNS;
        let page_count = matches.len().div_ceil(visible_items).max(1);
        self.jei_page = self.jei_page.min(page_count - 1);

        let detail = self.jei_selected;
        if let Some(detail) = detail {
            if Self::jei_button(ui, previous, false, true) {
                self.jei_selected = None;
            }
            Self::text(
                ui,
                ep::Pos2::new(panel.center().x, button_y + button_size / 2.0),
                "配方详情",
                16.0,
                egui::Color32::from_rgb(0xF0, 0xF0, 0xF0),
            );
            let _ = Self::jei_button(ui, next, true, false);
            self.draw_recipe_detail(ui, panel, detail);
        } else {
            if Self::jei_button(ui, previous, false, self.jei_page > 0) {
                self.jei_page -= 1;
            }
            if Self::jei_button(ui, next, true, self.jei_page + 1 < page_count) {
                self.jei_page += 1;
            }
            Self::text(
                ui,
                ep::Pos2::new(panel.center().x, button_y + button_size / 2.0),
                format!("物品 {}/{}", self.jei_page + 1, page_count),
                14.0,
                egui::Color32::from_rgb(0xF0, 0xF0, 0xF0),
            );

            let start = self.jei_page * visible_items;
            for (index, block) in matches
                .iter()
                .copied()
                .skip(start)
                .take(visible_items)
                .enumerate()
            {
                let row = index / JEI_COLUMNS;
                let column = index % JEI_COLUMNS;
                let rect = ep::Rect::from_min_size(
                    ep::Pos2::new(
                        grid_x + column as f32 * (slot + gap),
                        grid_top + row as f32 * (slot + gap),
                    ),
                    ep::Vec2::splat(slot),
                );
                self.draw_jei_item(ui, rect, block, start + index);
            }
            if matches.is_empty() {
                Self::text(
                    ui,
                    ep::Pos2::new(panel.center().x, grid_top + 42.0),
                    "没有找到物品",
                    13.0,
                    egui::Color32::from_rgb(0xB8, 0xB8, 0xB8),
                );
            }
        }

        let search_y = panel.max.y - 42.0;
        let search_rect = ep::Rect::from_min_max(
            ep::Pos2::new(panel.min.x + 5.0, search_y),
            ep::Pos2::new(panel.max.x - 43.0, panel.max.y - 6.0),
        );
        let options_rect = ep::Rect::from_min_max(
            ep::Pos2::new(panel.max.x - 39.0, search_y),
            ep::Pos2::new(panel.max.x - 5.0, panel.max.y - 6.0),
        );
        ui.painter()
            .rect_filled(search_rect, 0.0, egui::Color32::from_rgb(0x05, 0x06, 0x07));
        ui.painter().rect_stroke(
            search_rect,
            0.0,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(0x83, 0x87, 0x8A)),
            egui::StrokeKind::Inside,
        );
        let search_response = ui.put(
            search_rect.shrink2(ep::Vec2::new(5.0, 2.0)),
            egui::TextEdit::singleline(&mut self.jei_search)
                .frame(egui::Frame::NONE)
                .font(egui::FontId::proportional(13.0))
                .hint_text("搜索物品"),
        );
        if search_response.changed() {
            self.jei_page = 0;
        }
        if Self::jei_button(ui, options_rect, false, true) {
            self.jei_search.clear();
            self.jei_page = 0;
        }
        Self::draw_wrench_icon(
            ui.painter(),
            options_rect.center(),
            egui::Color32::from_rgb(0xBC, 0xBC, 0xBC),
        );
    }

    fn jei_button(ui: &mut egui::Ui, rect: ep::Rect, right: bool, enabled: bool) -> bool {
        let response = ui.allocate_rect(rect, egui::Sense::click());
        let fill = if !enabled {
            egui::Color32::from_rgb(0x28, 0x2A, 0x2C)
        } else if response.hovered() {
            egui::Color32::from_rgb(0x65, 0x67, 0x69)
        } else {
            egui::Color32::from_rgb(0x3B, 0x3D, 0x3F)
        };
        let painter = ui.painter();
        painter.rect_filled(rect, 0.0, fill);
        painter.rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(0x0A, 0x0B, 0x0C)),
            egui::StrokeKind::Inside,
        );
        painter.rect_stroke(
            rect.shrink(3.0),
            0.0,
            egui::Stroke::new(
                1.0,
                if enabled {
                    egui::Color32::from_rgb(0x86, 0x89, 0x8C)
                } else {
                    egui::Color32::from_rgb(0x4B, 0x4D, 0x4F)
                },
            ),
            egui::StrokeKind::Inside,
        );
        let center = rect.center();
        let points = if right {
            vec![
                ep::Pos2::new(center.x - 5.0, center.y - 9.0),
                ep::Pos2::new(center.x + 6.0, center.y),
                ep::Pos2::new(center.x - 5.0, center.y + 9.0),
            ]
        } else {
            vec![
                ep::Pos2::new(center.x + 5.0, center.y - 9.0),
                ep::Pos2::new(center.x - 6.0, center.y),
                ep::Pos2::new(center.x + 5.0, center.y + 9.0),
            ]
        };
        painter.add(ep::Shape::convex_polygon(
            points,
            if enabled {
                egui::Color32::from_rgb(0xE4, 0xE4, 0xE4)
            } else {
                egui::Color32::from_rgb(0x5C, 0x5C, 0x5C)
            },
            egui::Stroke::NONE,
        ));
        response.clicked() && enabled
    }

    fn draw_wrench_icon(painter: &egui::Painter, center: ep::Pos2, color: egui::Color32) {
        let stroke = egui::Stroke::new(2.0, color);
        painter.line_segment(
            [
                center + ep::Vec2::new(-6.0, -6.0),
                center + ep::Vec2::new(5.0, 5.0),
            ],
            stroke,
        );
        painter.circle_stroke(center + ep::Vec2::new(6.0, 6.0), 3.0, stroke);
        painter.circle_stroke(center + ep::Vec2::new(-6.0, -6.0), 4.0, stroke);
        painter.line_segment(
            [
                center + ep::Vec2::new(-9.0, -9.0),
                center + ep::Vec2::new(-4.0, -4.0),
            ],
            egui::Stroke::new(3.0, egui::Color32::from_rgb(0x14, 0x16, 0x18)),
        );
    }

    fn draw_jei_icon(&self, ui: &egui::Ui, rect: ep::Rect, block: Block, face: Face) {
        let tile = self.atlas.tile_id(block, block.tile(face));
        let (u0, v0, u1, v1) = self.atlas.uv(tile);
        ui.painter().image(
            self.atlas_texture.id(),
            rect,
            ep::Rect::from_min_max(ep::Pos2::new(u0, v0), ep::Pos2::new(u1, v1)),
            egui::Color32::WHITE,
        );
    }

    fn draw_jei_item(&mut self, ui: &mut egui::Ui, rect: ep::Rect, block: Block, index: usize) {
        let response = ui.allocate_rect(rect, egui::Sense::click());
        if response.hovered() {
            ui.painter().rect_filled(
                rect,
                0.0,
                egui::Color32::from_rgba_premultiplied(120, 120, 120, 90),
            );
        }
        let face = match index % 3 {
            0 => Face::Side,
            1 => Face::Top,
            _ => Face::Bottom,
        };
        self.draw_jei_icon(ui, rect.shrink(1.0), block, face);
        if response.hovered() {
            response.clone().on_hover_text(format!(
                "{}\n左键添加到物品栏\n右键查看配方与用途",
                block.def().display
            ));
        }
        if response.clicked_by(egui::PointerButton::Primary) {
            self.inventory.give(block);
        }
        if response.clicked_by(egui::PointerButton::Secondary) {
            self.jei_selected = Some(block);
        }
    }

    fn draw_inventory_slot(
        &mut self,
        ui: &mut egui::Ui,
        index: usize,
        rect: ep::Rect,
        show_count: bool,
    ) {
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
            if show_count {
                painter.text(
                    rect.right_bottom() - egui::Vec2::new(4.0, 3.0),
                    egui::Align2::RIGHT_BOTTOM,
                    stack.count.to_string(),
                    egui::FontId::proportional(12.0),
                    egui::Color32::WHITE,
                );
            }
            if hovered {
                response.clone().on_hover_text(stack.block.def().display);
            }
        }

        if response.clicked() {
            self.inventory.select(index);
        }
    }

    fn draw_debug(&self, ui: &egui::Ui) {
        let ctx = ui.ctx().clone();
        let screen_width = ctx.viewport_rect().width();
        let panel_width = (screen_width - 32.0).clamp(760.0, 1080.0);
        let column_width = ((panel_width - 40.0) / 2.0).max(170.0);
        egui::Area::new(egui::Id::new("mc-debug-overlay"))
            .order(egui::Order::Foreground)
            .fixed_pos(egui::pos2(8.0, 8.0))
            .interactable(false)
            // Keep the top-left anchor fixed even when the panel is taller than
            // a small window; otherwise egui can re-position it between frames.
            .constrain(false)
            .show(&ctx, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_rgba_premultiplied(4, 6, 10, 226))
                    .stroke(egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgb(0x72, 0x52, 0x2F),
                    ))
                    .corner_radius(4.0)
                    .inner_margin(8.0)
                    .show(ui, |ui| {
                        // A stable width plus truncating labels prevents the
                        // per-frame number changes from re-flowing the whole
                        // panel (the classic F3 HUD jitter).
                        ui.set_width(panel_width);
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                        ui.spacing_mut().item_spacing = egui::vec2(6.0, 3.0);
                        ui.horizontal_top(|ui| {
                            Self::debug_column(ui, column_width, |ui| {
                                self.draw_debug_perf(ui);
                            });
                            ui.add_space(10.0);
                            ui.separator();
                            ui.add_space(10.0);
                            Self::debug_column(ui, column_width, |ui| {
                                self.draw_debug_resources(ui);
                            });
                        });
                    });
            });
    }

    /// Fixed-width column used by the F3 overlay.
    ///
    /// `Ui::allocate_ui_with_layout` still sizes to its content, so the
    /// combination of a fixed width and `TextWrapMode::Truncate` is what keeps
    /// the layout from re-flowing as FPS, timings and counters change.
    fn debug_column<R>(
        ui: &mut egui::Ui,
        width: f32,
        add_contents: impl FnOnce(&mut egui::Ui) -> R,
    ) -> R {
        let response = ui.allocate_ui_with_layout(
            egui::vec2(width, 0.0),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                ui.set_width(width);
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                add_contents(ui)
            },
        );
        response.inner
    }

    /// Left F3 column: frame-time breakdown, sparkline and stage table.
    fn draw_debug_perf(&self, ui: &mut egui::Ui) {
        let p = &self.profiler;
        let total_ms = p.total.last_ms.max(0.0001);
        let fps = if p.total.last_ms > 0.0 {
            1000.0 / p.total.last_ms
        } else {
            0.0
        };
        let fps_avg = if p.total.avg_ms > 0.0 {
            1000.0 / p.total.avg_ms
        } else {
            0.0
        };
        let grey = egui::Color32::from_gray(0xBF);
        let value = egui::Color32::from_rgb(0xE0, 0xE0, 0xE0);

        ui.label(
            egui::RichText::new(format!("DEBUG  ·  {fps:.0} FPS  ·  {total_ms:.2} ms"))
                .strong()
                .color(egui::Color32::from_rgb(0xFF, 0xEE, 0x58)),
        );
        let cpu_ms = p.cpu_active_ms();
        ui.label(
            egui::RichText::new(format!(
                "帧 avg {:.2} / max {:.2} ms · 平均 {:.0} FPS · CPU 活动 {:.2} ms · Present {:.2} ms",
                p.total.avg_ms,
                p.total.max_ms,
                fps_avg,
                cpu_ms,
                p.present.last_ms
            ))
            .small()
            .color(grey),
        );
        let (bound, bound_color) = if cpu_ms >= total_ms * 0.7 && cpu_ms >= 1.0 {
            ("CPU 限制", egui::Color32::from_rgb(0xFF, 0x8A, 0x65))
        } else if p.present.last_ms >= total_ms * 0.5 {
            (
                if total_ms <= 20.0 {
                    "GPU / vsync 限制（帧率正常）"
                } else {
                    "GPU / vsync 限制"
                },
                egui::Color32::from_rgb(0xFF, 0xD1, 0x66),
            )
        } else {
            ("均衡 / 等待输入", egui::Color32::from_rgb(0x9C, 0xDC, 0xFE))
        };
        ui.label(
            egui::RichText::new(format!("主要瓶颈: {bound}"))
                .small()
                .strong()
                .color(bound_color),
        );

        let bar_width = ui.available_width().clamp(170.0, 360.0);
        let (graph_rect, _) =
            ui.allocate_exact_size(egui::vec2(bar_width, 34.0), egui::Sense::hover());
        Self::draw_debug_frame_graph(ui.painter(), graph_rect, &p.frame_history_ms);

        const C_PHYSICS: egui::Color32 = egui::Color32::from_rgb(0x4F, 0xC3, 0xF7);
        const C_RAYCAST: egui::Color32 = egui::Color32::from_rgb(0x81, 0xC7, 0x84);
        const C_MESH: egui::Color32 = egui::Color32::from_rgb(0xFF, 0xB7, 0x4D);
        const C_HIGHLIGHT: egui::Color32 = egui::Color32::from_rgb(0xBA, 0x68, 0xC8);
        const C_UI: egui::Color32 = egui::Color32::from_rgb(0xF0, 0x62, 0x92);
        const C_TESS: egui::Color32 = egui::Color32::from_rgb(0x4D, 0xD0, 0xE1);
        const C_ACQUIRE: egui::Color32 = egui::Color32::from_rgb(0xFF, 0xF1, 0x76);
        const C_SUBMIT: egui::Color32 = egui::Color32::from_rgb(0xA1, 0x88, 0x7F);
        const C_PRESENT: egui::Color32 = egui::Color32::from_rgb(0xE5, 0x73, 0x73);

        let cpu_entries: [(&str, StageTiming, egui::Color32); 8] = [
            ("物理", p.physics, C_PHYSICS),
            ("射线", p.raycast, C_RAYCAST),
            ("网格", p.mesh, C_MESH),
            ("高亮", p.highlight, C_HIGHLIGHT),
            ("UI", p.ui, C_UI),
            ("Tess", p.tessellate, C_TESS),
            ("取帧", p.acquire, C_ACQUIRE),
            ("提交", p.submit, C_SUBMIT),
        ];
        let mut share_entries: Vec<(&str, f64, egui::Color32)> = cpu_entries
            .iter()
            .map(|(name, timing, color)| (*name, timing.last_ms, *color))
            .collect();
        share_entries.push(("Present", p.present.last_ms, C_PRESENT));

        let (bar_rect, _) =
            ui.allocate_exact_size(egui::vec2(bar_width, 12.0), egui::Sense::hover());
        Self::draw_debug_stage_bar(ui.painter(), bar_rect, &share_entries, total_ms);

        ui.add_space(2.0);
        egui::Grid::new("mc-debug-perf-grid")
            .num_columns(5)
            .spacing([10.0, 2.0])
            .show(ui, |ui| {
                for header in ["阶段", "last", "avg", "max", "占比"] {
                    ui.label(egui::RichText::new(header).small().color(grey));
                }
                ui.end_row();

                for &(name, timing, color) in &cpu_entries {
                    let stress = Self::debug_stress_color(timing.last_ms, total_ms);
                    ui.label(egui::RichText::new(name).monospace().color(color));
                    ui.label(
                        egui::RichText::new(format!("{:6.2}", timing.last_ms))
                            .monospace()
                            .color(stress),
                    );
                    ui.label(
                        egui::RichText::new(format!("{:6.2}", timing.avg_ms))
                            .monospace()
                            .color(stress),
                    );
                    ui.label(
                        egui::RichText::new(format!("{:6.2}", timing.max_ms))
                            .monospace()
                            .color(grey),
                    );
                    let share = timing.last_ms / total_ms * 100.0;
                    ui.label(
                        egui::RichText::new(format!("{share:5.1}%"))
                            .monospace()
                            .color(stress),
                    );
                    ui.end_row();
                }

                // Total and present are not part of the CPU-share bar, but the
                // table should still show where the rest of the frame goes.
                let present_stress = Self::debug_stress_color(p.present.last_ms, total_ms);
                ui.label(egui::RichText::new("Present").monospace().color(C_PRESENT));
                ui.label(
                    egui::RichText::new(format!("{:6.2}", p.present.last_ms))
                        .monospace()
                        .color(present_stress),
                );
                ui.label(egui::RichText::new(format!("{:6.2}", p.present.avg_ms)).monospace());
                ui.label(egui::RichText::new(format!("{:6.2}", p.present.max_ms)).monospace());
                ui.label(
                    egui::RichText::new(format!("{:5.1}%", p.present.last_ms / total_ms * 100.0))
                        .monospace()
                        .color(present_stress),
                );
                ui.end_row();

                ui.label(egui::RichText::new("总计").monospace().strong());
                ui.label(
                    egui::RichText::new(format!("{:6.2}", p.total.last_ms))
                        .monospace()
                        .strong(),
                );
                ui.label(
                    egui::RichText::new(format!("{:6.2}", p.total.avg_ms))
                        .monospace()
                        .strong(),
                );
                ui.label(
                    egui::RichText::new(format!("{:6.2}", p.total.max_ms))
                        .monospace()
                        .strong(),
                );
                ui.label("");
                ui.end_row();
            });

        let remaining = (total_ms - p.cpu_active_ms()).max(0.0);
        ui.label(
            egui::RichText::new(format!(
                "物理 tick {} · UI shapes {} · clipped {} · 等待/空闲 {:.2} ms",
                p.physics_ticks, p.ui_shapes, p.clipped_primitives, remaining
            ))
            .small()
            .color(value),
        );
    }

    /// Right F3 column: GPU, mesher, world and system resource numbers.
    fn draw_debug_resources(&self, ui: &mut egui::Ui) {
        let grey = egui::Color32::from_gray(0xBF);
        let value = egui::Color32::from_rgb(0xE0, 0xE0, 0xE0);
        let heading = egui::Color32::from_rgb(0xFF, 0xCC, 0x80);
        let good = egui::Color32::from_rgb(0x9C, 0xDC, 0xFE);
        let warn = egui::Color32::from_rgb(0xFF, 0x8A, 0x65);

        let g = self.gpu_debug;
        ui.label(egui::RichText::new("GPU").strong().color(heading));
        ui.label(egui::RichText::new(&self.gpu_info).small().color(grey));
        ui.label(
            egui::RichText::new(format!(
                "{}x{} @ {:.2}x · 显存估算 {}",
                g.surface_width,
                g.surface_height,
                ui.ctx().pixels_per_point(),
                Self::debug_bytes(g.total_gpu_bytes())
            ))
            .small()
            .color(value),
        );
        ui.label(
            egui::RichText::new(format!(
                "地形缓冲 {} · 透明缓冲 {} · 场景/深度 {} · 阴影 {} · atlas {}",
                Self::debug_bytes(g.vertex_buffer_bytes + g.index_buffer_bytes),
                Self::debug_bytes(
                    g.transparent_vertex_buffer_bytes + g.transparent_index_buffer_bytes
                ),
                Self::debug_bytes(g.scene_texture_bytes + g.depth_texture_bytes),
                Self::debug_bytes(g.shadow_texture_bytes),
                Self::debug_bytes(g.atlas_texture_bytes)
            ))
            .small()
            .color(grey),
        );
        ui.label(
            egui::RichText::new(format!(
                "不透明 V {} · I {} · △ {}    透明 V {} · I {} · △ {}",
                g.vertex_count,
                g.index_count,
                g.opaque_triangles(),
                g.transparent_vertex_count,
                g.transparent_index_count,
                g.transparent_triangles()
            ))
            .small()
            .color(value),
        );

        ui.label(
            egui::RichText::new(format!(
                "GPU 提交 主场景 {} 三角 / {} draws · 阴影 {} 三角 / {} draws · 上次上传 {}",
                g.visible_indices / 3,
                g.scene_draw_calls,
                g.shadow_indices / 3,
                g.shadow_draw_calls,
                Self::debug_bytes(g.last_mesh_upload_bytes),
            ))
            .small()
            .color(grey),
        );

        ui.add_space(2.0);
        ui.label(egui::RichText::new("网格 worker").strong().color(heading));
        let mesh_color = if self.profiler.mesh_build_ms >= 16.7 {
            warn
        } else if self.profiler.mesh_build_ms >= 8.0 {
            egui::Color32::from_rgb(0xFF, 0xD1, 0x66)
        } else {
            value
        };
        ui.label(
            egui::RichText::new(format!(
                "build {:.2} ms · upload {:.2} ms · 队列 {} · 最近延迟 {:.0} ms",
                self.profiler.mesh_build_ms,
                self.profiler.mesh_upload_ms,
                self.profiler.mesh_pending,
                self.profiler.mesh_request_latency_ms
            ))
            .small()
            .color(mesh_color),
        );
        ui.label(
            egui::RichText::new(format!(
                "请求中心 {:?} · 已加载中心 {:?} · 重建距离 {}",
                self.mesh_requested_center,
                self.mesh_center,
                crate::app::MESH_REBUILD_DISTANCE
            ))
            .small()
            .color(grey),
        );

        if self.screen == LauncherScreen::InWorld {
            ui.add_space(2.0);
            ui.label(egui::RichText::new("世界").strong().color(heading));
            if let Some(world) = self.physics_world.as_ref() {
                let cached = world.cached_column_count();
                let edits = world.edits().count();
                ui.label(
                    egui::RichText::new(format!(
                        "缓存列 {cached} · 编辑 {edits} · 水下 {}",
                        if self.profiler.underwater {
                            "是"
                        } else {
                            "否"
                        }
                    ))
                    .small()
                    .color(value),
                );
            }
            if let Some(player) = self.player.as_ref() {
                let x = player.position.x;
                let y = player.position.y;
                let z = player.position.z;
                let block_x = x.floor() as i64;
                let block_z = z.floor() as i64;
                let cx = block_x.div_euclid(CHUNK_SIZE);
                let cz = block_z.div_euclid(CHUNK_SIZE);
                let surface = self
                    .physics_world
                    .as_ref()
                    .map(|world| world.surface_top(block_x, block_z))
                    .unwrap_or(0);
                ui.label(
                    egui::RichText::new(format!(
                        "玩家 ({x:.1}, {y:.1}, {z:.1}) · 区块 ({cx}, {cz}) · 地表 y={surface}"
                    ))
                    .small()
                    .color(value),
                );
                ui.label(
                    egui::RichText::new(format!(
                        "速度 ({:.2}, {:.2}, {:.2}) · {} · yaw {:.0}° pitch {:.0}°",
                        player.velocity.x,
                        player.velocity.y,
                        player.velocity.z,
                        if player.on_ground { "着地" } else { "空中" },
                        self.camera.yaw.to_degrees(),
                        self.camera.pitch.to_degrees()
                    ))
                    .small()
                    .color(grey),
                );
            }
            let target = self
                .target_block
                .map(|target| {
                    format!(
                        "({}, {}, {})",
                        target.block.0, target.block.1, target.block.2
                    )
                })
                .unwrap_or_else(|| "无".to_owned());
            let target_name = self
                .target_block_kind
                .map(|block| block.def().display)
                .unwrap_or("-");
            ui.label(
                egui::RichText::new(format!(
                    "准星 {target_name} {target} · {} · {}",
                    self.game_mode.display(),
                    if self.inventory.open {
                        "物品栏打开"
                    } else {
                        "物品栏关闭"
                    }
                ))
                .small()
                .color(value),
            );
            ui.label(
                egui::RichText::new(format!(
                    "视距 {} 格 · LOD 近 {} 格 · chunk {} · 重建阈值 {}",
                    crate::app::MESH_RADIUS,
                    LOD_NEAR_RADIUS,
                    CHUNK_SIZE,
                    crate::app::MESH_REBUILD_DISTANCE
                ))
                .small()
                .color(grey),
            );
        }

        ui.add_space(2.0);
        ui.label(egui::RichText::new("系统").strong().color(heading));
        let system = self.system_stats;
        if system.rss_bytes > 0 {
            ui.label(
                egui::RichText::new(format!(
                    "RSS {} · 峰值 {} · 线程 {}",
                    Self::debug_bytes(system.rss_bytes),
                    Self::debug_bytes(system.peak_rss_bytes),
                    system.thread_count
                ))
                .small()
                .color(value),
            );
        } else {
            ui.label(
                egui::RichText::new("RSS/线程: 仅 Linux 可采样")
                    .small()
                    .color(grey),
            );
        }
        let cpu_color = if system.cpu_percent >= 90.0 {
            warn
        } else if system.cpu_percent >= 50.0 {
            egui::Color32::from_rgb(0xFF, 0xD1, 0x66)
        } else {
            good
        };
        ui.label(
            egui::RichText::new(format!(
                "进程 CPU {:.0}% (单核百分比) · 负载 {:.2}",
                system.cpu_percent, system.load_avg_1
            ))
            .small()
            .color(cpu_color),
        );
    }

    fn draw_debug_frame_graph(painter: &egui::Painter, rect: egui::Rect, history: &[f32]) {
        painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(0x10, 0x14, 0x1A));
        if history.is_empty() {
            return;
        }
        let max_ms = history
            .iter()
            .fold(33.3_f32, |acc, value| acc.max(*value))
            .max(33.3);
        let budget_y = rect.bottom() - (16.67 / max_ms).clamp(0.0, 1.0) * rect.height();
        painter.line_segment(
            [
                egui::pos2(rect.left(), budget_y),
                egui::pos2(rect.right(), budget_y),
            ],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(0x6B, 0x8E, 0x3F)),
        );

        let last = history.len().saturating_sub(1).max(1) as f32;
        let points: Vec<egui::Pos2> = history
            .iter()
            .enumerate()
            .map(|(index, ms)| {
                let x = rect.left() + rect.width() * index as f32 / last;
                let y = rect.bottom() - (*ms / max_ms).clamp(0.0, 1.0) * rect.height();
                egui::pos2(x, y)
            })
            .collect();
        let latest = history.last().copied().unwrap_or(0.0);
        let color = if latest > 16.67 {
            egui::Color32::from_rgb(0xFF, 0x8A, 0x65)
        } else {
            egui::Color32::from_rgb(0x81, 0xC7, 0x84)
        };
        painter.add(egui::Shape::line(points, egui::Stroke::new(1.2, color)));
        painter.text(
            rect.right_top() + egui::vec2(-4.0, 2.0),
            egui::Align2::RIGHT_TOP,
            format!("{max_ms:.0}ms"),
            egui::FontId::monospace(10.0),
            egui::Color32::from_gray(0x9A),
        );
    }

    fn draw_debug_stage_bar(
        painter: &egui::Painter,
        rect: egui::Rect,
        entries: &[(&str, f64, egui::Color32)],
        total_ms: f64,
    ) {
        painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(0x22, 0x26, 0x2B));
        let sum: f64 = entries.iter().map(|(_, ms, _)| *ms).sum();
        let scale_total = total_ms.max(sum).max(0.0001);
        let mut x = rect.left();
        for (_, ms, color) in entries {
            if *ms <= 0.0 {
                continue;
            }
            let width = (ms / scale_total) * rect.width() as f64;
            let width = width.max(0.5) as f32;
            let segment = egui::Rect::from_min_size(
                egui::pos2(x, rect.top()),
                egui::vec2(width.min(rect.right() - x), rect.height()),
            );
            painter.rect_filled(segment, 1.0, *color);
            x += width;
            if x >= rect.right() {
                break;
            }
        }
    }

    fn debug_stress_color(ms: f64, total_ms: f64) -> egui::Color32 {
        let share = if total_ms > 0.0 { ms / total_ms } else { 0.0 };
        if ms >= 8.0 || share >= 0.45 {
            egui::Color32::from_rgb(0xFF, 0x6B, 0x6B)
        } else if ms >= 4.0 || share >= 0.25 {
            egui::Color32::from_rgb(0xFF, 0xB7, 0x4D)
        } else if ms >= 1.5 || share >= 0.10 {
            egui::Color32::from_rgb(0xFF, 0xF1, 0x76)
        } else {
            egui::Color32::from_rgb(0x9C, 0xDC, 0xFE)
        }
    }

    fn debug_bytes(bytes: u64) -> String {
        const KIB: f64 = 1024.0;
        const MIB: f64 = 1024.0 * 1024.0;
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        let bytes = bytes as f64;
        if bytes >= GIB {
            format!("{:.2} GB", bytes / GIB)
        } else if bytes >= MIB {
            format!("{:.2} MB", bytes / MIB)
        } else if bytes >= KIB {
            format!("{:.1} KB", bytes / KIB)
        } else {
            format!("{bytes:.0} B")
        }
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
        let terrain = heightmap.sample(info.spawn_x, info.spawn_z);
        let column = column_gen.generate_with_terrain(info.spawn_x, info.spawn_z, terrain);

        assert_eq!(safe_spawn_y(&column), Some(info.spawn_height));
        assert!(terrain.height > heightmap.sea_level());
    }
}
