//! CPU 体素面剔除与网格生成。
//!
//! 当前先按玩家附近的方块生成 camera-relative 网格。网格只包含暴露面，
//! 地下内部面在 CPU 端剔除；区块网格按 LOD 缓存，合并后交给 GPU。

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use glam::DVec3;

use crate::render::probe::ProbeVolume;
use crate::world::atlas::Atlas;
use crate::world::block::{self, Block, Face, LIGHT_MAX};
use crate::world::column::{Column, Y_MAX, Y_MIN};
use crate::world::light;
use crate::world::voxel::{GeneratedVoxelWorld, VoxelWorld};

/// Minecraft-style fluid surface height. Water occupies a voxel for world
/// storage and placement, but its visible surface sits slightly below the
/// voxel boundary instead of looking like a full solid cube.
const WATER_SURFACE_HEIGHT: f32 = 0.875;

/// Chunk edge length in blocks. 32 divides both LOD steps (2 and 4), so a
/// chunk always contains an integer number of LOD cells.
pub const CHUNK_SIZE: i64 = 32;
/// Requested view distance in blocks.
pub const VIEW_RADIUS: i64 = 192;
/// Chunk radius needed to cover `VIEW_RADIUS`. A 13x13 chunk square covers
/// 192 blocks even from the worst position inside the centre chunk.
pub const CHUNK_VIEW_RADIUS: i64 = (VIEW_RADIUS + CHUNK_SIZE - 1) / CHUNK_SIZE;
/// Full-resolution chunk LOD boundary.
pub const LOD_NEAR_RADIUS: i64 = 64;
/// Mid-ring outer boundary. Mid chunks are sampled every [`LOD_MID_STEP`]
/// blocks.
pub const LOD_MID_RADIUS: i64 = 128;
pub const LOD_MID_STEP: i64 = 2;
/// Far-ring sample step.
pub const LOD_FAR_STEP: i64 = 4;
/// Horizontal halo (in blocks) needed for correct sky-light flood fill at a
/// chunk border. Light travels at most `LIGHT_MAX` blocks, so the mesher
/// snapshots this much extra terrain around every near chunk.
const LIGHT_HALO: i64 = LIGHT_MAX as i64 + 1;
/// Downward skirt for opaque LOD cells that need to hide a possible crack
/// at a chunk boundary or at the outer mesh edge.  Fluids do not use it;
/// their transparent side geometry reads as visible walls.
const LOD_SKIRT_HEIGHT: f32 = 8.0;

/// GPU 顶点：位置、atlas UV、切线空间、环境遮蔽值，以及透明材质标记。
///
/// 每个方块面仍然使用独立的四个顶点，因此法线保持体素世界需要的硬边；
/// AO 则可以在一个面内平滑插值，避免墙角看起来像贴了一层固定深色。
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub normal: [f32; 3],
    pub ao: f32,
    /// 0 = 普通不透明/alpha-test，1 = 流体，2 = 树叶 cutout。
    /// 透明网格中的玻璃仍保持 0。
    pub material: f32,
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
    /// Face-local texture coordinates, independent of atlas packing.
    pub local_uv: [f32; 2],
    /// Neighbor material tiles at U-, U+, V-, V+ for edge blending.
    pub blend_tiles: [u32; 4],
    /// x/y = selected subregion, z/w = region columns/rows.
    pub texture_region: [u32; 4],
    /// 天空可见度 `0..=1`：洪泛光照按顶点平滑后的结果。必须排在最后,
    /// 与 [`Vertex::desc`] 里追加到末尾的 location 10 保持一致的字段顺序.
    pub light: f32,
}

impl Vertex {
    pub const fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 11] = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x2,
            2 => Float32x3,
            3 => Float32,
            4 => Float32,
            5 => Float32x3,
            6 => Float32x3,
            7 => Float32x2,
            8 => Uint32x4,
            9 => Uint32x4,
            10 => Float32
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

#[derive(Default, Debug)]
pub struct VoxelMesh {
    /// 不透明与 alpha-test 方块，先绘制以建立可靠的深度缓冲。
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// 玻璃和水，后绘制且不写深度，保证能看到后面的地形。
    pub transparent_vertices: Vec<Vertex>,
    pub transparent_indices: Vec<u32>,
}

type BlockEdit = ((i64, i64, i64), Block);

/// Stable identity for a 32x32 column chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkKey {
    pub cx: i32,
    pub cz: i32,
}

impl ChunkKey {
    pub fn from_block(x: i64, z: i64) -> Self {
        Self {
            cx: x.div_euclid(CHUNK_SIZE) as i32,
            cz: z.div_euclid(CHUNK_SIZE) as i32,
        }
    }

    pub fn min_x(self) -> i64 {
        self.cx as i64 * CHUNK_SIZE
    }

    pub fn min_z(self) -> i64 {
        self.cz as i64 * CHUNK_SIZE
    }

    pub fn max_x(self) -> i64 {
        self.min_x() + CHUNK_SIZE - 1
    }

    pub fn max_z(self) -> i64 {
        self.min_z() + CHUNK_SIZE - 1
    }

    pub fn center(self) -> (i64, i64) {
        (self.min_x() + CHUNK_SIZE / 2, self.min_z() + CHUNK_SIZE / 2)
    }

    /// Chebyshev distance from a block position to the nearest point in the
    /// chunk. This drives the LOD level so a chunk's corners are not treated
    /// as if they were farther away than its closest face.
    pub fn distance_to(self, x: i64, z: i64) -> i64 {
        let dx = if x < self.min_x() {
            self.min_x() - x
        } else if x > self.max_x() {
            x - self.max_x()
        } else {
            0
        };
        let dz = if z < self.min_z() {
            self.min_z() - z
        } else if z > self.max_z() {
            z - self.max_z()
        } else {
            0
        };
        dx.max(dz)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LodLevel {
    Near,
    Mid,
    Far,
}

impl LodLevel {
    pub fn step(self) -> i64 {
        match self {
            Self::Near => 1,
            Self::Mid => LOD_MID_STEP,
            Self::Far => LOD_FAR_STEP,
        }
    }

    pub fn from_distance(distance: i64) -> Self {
        if distance <= LOD_NEAR_RADIUS {
            Self::Near
        } else if distance <= LOD_MID_RADIUS {
            Self::Mid
        } else {
            Self::Far
        }
    }
}

/// 兼容旧 worker 的整片网格请求参数。
pub struct MeshBuildInput<'a> {
    pub seed: u64,
    pub center: (i64, i64),
    pub radius: i64,
    pub origin: DVec3,
    pub atlas: &'a Atlas,
    pub initial_columns: &'a [Column],
    pub edits: &'a [BlockEdit],
}

struct MeshRequest {
    kind: MeshRequestKind,
    origin: DVec3,
    edits: Vec<BlockEdit>,
}

enum MeshRequestKind {
    Region {
        request_id: u64,
        center: (i64, i64),
        radius: i64,
    },
    Chunk {
        key: ChunkKey,
        lod: LodLevel,
    },
}

pub struct MeshResult {
    pub request_id: u64,
    pub center: (i64, i64),
    pub key: ChunkKey,
    pub lod: LodLevel,
    pub origin: DVec3,
    pub mesh: VoxelMesh,
    /// 世界空间辐照度探针; 只有整片 Region 请求才烘焙, 增量 chunk 为 None.
    pub probes: Option<ProbeVolume>,
    /// CPU time the worker spent building this result. This excludes time
    /// waiting in the request queue, so it is the mesher's real hot-path cost.
    pub build_time: Duration,
}

/// Background mesher.
///
/// The incremental API builds one chunk per request (`request_chunk` / `try_recv`).
/// The app uses `request` / `try_take_latest`: reuse unchanged CPU chunk meshes,
/// then assemble one camera-relative GPU mesh for the existing renderer.
pub struct VoxelMeshWorker {
    requests: mpsc::Sender<MeshRequest>,
    results: mpsc::Receiver<MeshResult>,
    next_request_id: u64,
    latest_request_id: Option<u64>,
    /// Requests sent minus results drained. Useful for exposing worker backlog
    /// in the debug overlay without blocking the render thread.
    pending_requests: usize,
}

impl VoxelMeshWorker {
    pub fn new(seed: u64, atlas: Arc<Atlas>, initial_columns: Vec<Column>) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<MeshRequest>();
        let (result_tx, result_rx) = mpsc::channel::<MeshResult>();

        thread::spawn(move || {
            let mut world = GeneratedVoxelWorld::with_columns(seed, initial_columns);
            let mut cache = RegionMeshCache::default();
            while let Ok(request) = request_rx.recv() {
                match request.kind {
                    MeshRequestKind::Region {
                        request_id,
                        center,
                        radius,
                    } => {
                        let build_start = std::time::Instant::now();
                        let mesh = cache.build(
                            &mut world,
                            center,
                            radius,
                            request.origin,
                            &atlas,
                            &request.edits,
                        );
                        // 探针烘焙复用同一份确定性世界与光照缓存, 因此能看到
                        // 地形 / 树冠 / 水体 / 玩家编辑; 埋在地下的探针会被跳过.
                        let probes = Some(ProbeVolume::bake(&mut world, center));
                        let build_time = build_start.elapsed();
                        let key = ChunkKey::from_block(center.0, center.1);
                        if result_tx
                            .send(MeshResult {
                                request_id,
                                center,
                                key,
                                lod: LodLevel::Near,
                                origin: request.origin,
                                mesh,
                                probes,
                                build_time,
                            })
                            .is_err()
                        {
                            break;
                        }
                        let (center_x, center_z) = key.center();
                        world.retain_columns_within(
                            center_x,
                            center_z,
                            VIEW_RADIUS + CHUNK_SIZE * 2,
                        );
                    }
                    MeshRequestKind::Chunk { key, lod } => {
                        let build_start = std::time::Instant::now();
                        let mesh = VoxelMesh::build_chunk(
                            &mut world,
                            key,
                            lod,
                            request.origin,
                            &atlas,
                            &request.edits,
                        );
                        let build_time = build_start.elapsed();
                        if result_tx
                            .send(MeshResult {
                                request_id: 0,
                                center: key.center(),
                                key,
                                lod,
                                origin: request.origin,
                                mesh,
                                probes: None,
                                build_time,
                            })
                            .is_err()
                        {
                            break;
                        }
                        let (center_x, center_z) = key.center();
                        world.retain_columns_within(
                            center_x,
                            center_z,
                            VIEW_RADIUS + CHUNK_SIZE * 2,
                        );
                    }
                }
            }
        });

        Self {
            requests: request_tx,
            results: result_rx,
            next_request_id: 0,
            latest_request_id: None,
            pending_requests: 0,
        }
    }

    /// Legacy combined-region request used by the current app.
    pub fn request(
        &mut self,
        center: (i64, i64),
        radius: i64,
        origin: DVec3,
        edits: Vec<BlockEdit>,
    ) {
        self.next_request_id = self.next_request_id.wrapping_add(1);
        let request_id = self.next_request_id;
        if self
            .requests
            .send(MeshRequest {
                kind: MeshRequestKind::Region {
                    request_id,
                    center,
                    radius,
                },
                origin,
                edits,
            })
            .is_ok()
        {
            self.latest_request_id = Some(request_id);
            self.pending_requests = self.pending_requests.saturating_add(1);
        }
    }

    /// Number of requests that have been sent but whose results have not been
    /// drained yet. Includes stale combined-region builds.
    pub fn pending_requests(&self) -> usize {
        self.pending_requests
    }

    /// Legacy: return only the newest combined-region result.
    pub fn try_take_latest(&mut self) -> Option<MeshResult> {
        let expected_id = self.latest_request_id?;
        let mut latest = None;
        while let Ok(result) = self.results.try_recv() {
            self.pending_requests = self.pending_requests.saturating_sub(1);
            if result.request_id == expected_id {
                latest = Some(result);
            }
        }
        if latest.is_some() {
            self.latest_request_id = None;
        }
        latest
    }

    /// Incremental chunk request for the future streaming renderer.
    pub fn request_chunk(
        &mut self,
        key: ChunkKey,
        lod: LodLevel,
        origin: DVec3,
        edits: Vec<BlockEdit>,
    ) {
        if self
            .requests
            .send(MeshRequest {
                kind: MeshRequestKind::Chunk { key, lod },
                origin,
                edits,
            })
            .is_ok()
        {
            self.pending_requests = self.pending_requests.saturating_add(1);
        }
    }

    /// Non-blocking receive of the next chunk result.
    pub fn try_recv(&mut self) -> Option<MeshResult> {
        let result = self.results.try_recv().ok();
        if result.is_some() {
            self.pending_requests = self.pending_requests.saturating_sub(1);
        }
        result
    }
}

impl VoxelMesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() && self.transparent_indices.is_empty()
    }

    /// 生成玩家附近的方块网格。
    ///
    /// `origin` 是 camera-relative 原点；权威世界坐标仍然使用 i64/f64，
    /// 只有送入 GPU 的顶点转换为相对 f32。
    pub fn build(
        world: &mut GeneratedVoxelWorld,
        center_x: i64,
        center_z: i64,
        radius: i64,
        origin: DVec3,
        atlas: &Atlas,
    ) -> Self {
        Self::build_region(
            world,
            center_x - radius,
            center_x + radius,
            center_z - radius,
            center_z + radius,
            origin,
            atlas,
        )
    }

    /// Build one independent chunk at the requested LOD.
    ///
    /// Near chunks use the full voxel mesher; mid/far chunks use a coarse
    /// heightfield sampled every `lod.step()` blocks. The worker asks for one
    /// chunk at a time, so results can be uploaded incrementally.
    pub fn build_chunk(
        world: &mut GeneratedVoxelWorld,
        key: ChunkKey,
        lod: LodLevel,
        origin: DVec3,
        atlas: &Atlas,
        edits: &[BlockEdit],
    ) -> Self {
        if !edits.is_empty() {
            world.apply_edits(edits.iter().map(|&((x, y, z), block)| (x, y, z, block)));
        }
        match lod {
            LodLevel::Near => Self::build_near_chunk(world, key, origin, atlas),
            LodLevel::Mid => Self::build_lod_region(world, key, LOD_MID_STEP, origin, atlas),
            LodLevel::Far => Self::build_lod_region(world, key, LOD_FAR_STEP, origin, atlas),
        }
    }

    /// Legacy combined-region builder.
    ///
    /// Small regions use the exact full-resolution mesher so existing callers
    /// and tests keep the same face counts. Larger regions are assembled from
    /// independent chunks, with mid/far chunks using the coarse LOD path.
    #[cfg(test)]
    fn build_parallel(input: MeshBuildInput<'_>) -> Self {
        let mut world =
            GeneratedVoxelWorld::with_columns(input.seed, input.initial_columns.to_vec());
        world.apply_edits(
            input
                .edits
                .iter()
                .map(|&((x, y, z), block)| (x, y, z, block)),
        );

        if input.radius <= CHUNK_SIZE {
            return Self::build_region(
                &mut world,
                input.center.0 - input.radius,
                input.center.0 + input.radius,
                input.center.1 - input.radius,
                input.center.1 + input.radius,
                input.origin,
                input.atlas,
            );
        }

        let start =
            ChunkKey::from_block(input.center.0 - input.radius, input.center.1 - input.radius);
        let end =
            ChunkKey::from_block(input.center.0 + input.radius, input.center.1 + input.radius);
        let mut mesh = Self::default();
        for cz in start.cz..=end.cz {
            for cx in start.cx..=end.cx {
                let key = ChunkKey { cx, cz };
                let lod = LodLevel::from_distance(key.distance_to(input.center.0, input.center.1));
                let part =
                    Self::build_chunk(&mut world, key, lod, input.origin, input.atlas, input.edits);
                append_mesh(&mut mesh, part);
            }
        }
        mesh
    }

    /// Diagnostic LOD-ring builder used by the renderer tests.
    #[cfg(test)]
    fn build_lod_rings(input: MeshBuildInput<'_>) -> Self {
        let mut world =
            GeneratedVoxelWorld::with_columns(input.seed, input.initial_columns.to_vec());
        let cells = lod_ring_cells(input.center, input.radius, LOD_NEAR_RADIUS, LOD_MID_STEP);
        let mut mesh = Self::default();

        for (x, z, step) in cells {
            let Some((y, block)) = world.terrain_surface_block(x, z) else {
                continue;
            };
            let surface = lod_surface_from_block(y, block);
            let base = DVec3::new(x as f64, 0.0, z as f64) - input.origin;
            let bx = base.x as f32;
            let bz = base.z as f32;
            let top = (surface.top as f64 - input.origin.y) as f32;
            let step_f = step as f32;
            append_lod_quad(
                &mut mesh,
                input.atlas,
                [
                    [bx, top, bz],
                    [bx + step_f, top, bz],
                    [bx + step_f, top, bz + step_f],
                    [bx, top, bz + step_f],
                ],
                [0.0, 1.0, 0.0],
                surface.block,
                Face::Top,
                x,
                surface.y,
                z,
            );
        }

        mesh
    }

    fn build_near_chunk(
        world: &mut GeneratedVoxelWorld,
        key: ChunkKey,
        origin: DVec3,
        atlas: &Atlas,
    ) -> Self {
        Self::build_region(
            world,
            key.min_x(),
            key.max_x(),
            key.min_z(),
            key.max_z(),
            origin,
            atlas,
        )
    }

    fn build_lod_region(
        world: &mut GeneratedVoxelWorld,
        key: ChunkKey,
        step: i64,
        origin: DVec3,
        atlas: &Atlas,
    ) -> Self {
        debug_assert!(step > 1, "near chunks use build_near_chunk");
        let min_x = key.min_x();
        let max_x = key.max_x();
        let min_z = key.min_z();
        let max_z = key.max_z();

        // Sample one cell outside the chunk as well, so side faces at the
        // chunk border see the real neighbour surface instead of an
        // artificial skirt.
        let sample_min_x = min_x - step;
        let sample_max_x = max_x + step;
        let sample_min_z = min_z - step;
        let sample_max_z = max_z + step;
        let mut surfaces: HashMap<(i64, i64), LodSurface> = HashMap::new();
        let mut z = sample_min_z;
        while z <= sample_max_z {
            let mut x = sample_min_x;
            while x <= sample_max_x {
                if let Some((y, block)) = world.terrain_surface_block(x, z) {
                    surfaces.insert((x, z), lod_surface_from_block(y, block));
                }
                x += step;
            }
            z += step;
        }

        // A mixed cell and its immediate ring are rebuilt at full resolution.
        // The extra ring means the fine band ends on a cell that is already
        // homogeneous by the four-corner test, reducing visible LOD cracks at
        // the transition back to coarse quads.
        let mut fine_cells: HashSet<(i64, i64)> = HashSet::new();
        let mut z = min_z;
        while z <= max_z {
            let mut x = min_x;
            while x <= max_x {
                if surfaces.contains_key(&(x, z)) && lod_cell_is_mixed(&surfaces, x, z, step) {
                    for (nx, nz) in [
                        (x, z),
                        (x - step, z),
                        (x + step, z),
                        (x, z - step),
                        (x, z + step),
                    ] {
                        if nx >= min_x && nx <= max_x && nz >= min_z && nz <= max_z {
                            fine_cells.insert((nx, nz));
                        }
                    }
                }
                x += step;
            }
            z += step;
        }

        let mut mesh = Self::default();
        let mut z = min_z;
        while z <= max_z {
            let mut x = min_x;
            while x <= max_x {
                if let Some(&surface) = surfaces.get(&(x, z)) {
                    if fine_cells.contains(&(x, z)) {
                        append_lod_fine_cell(&mut mesh, world, origin, atlas, x, z, step);
                    } else {
                        let boundary =
                            x == min_x || x + step > max_x || z == min_z || z + step > max_z;
                        append_lod_cell(
                            &mut mesh, origin, atlas, x, z, step, surface, &surfaces, boundary,
                        );
                    }
                }
                x += step;
            }
            z += step;
        }

        // 树冠不能参与 LOD 高度场：否则它会被当成“这一格的地表”，
        // 在 step×step 的尺度上拉成整片绿色墙。这里把生成器产生的树
        // 按真实方块重新提交；树是稀疏的，全分辨率代价可控，并且保留
        // 了 MC 式树干和 alpha-cutout 树叶形状。
        let tree_blocks = world.tree_blocks_in_region(min_x - 2, max_x + 3, min_z - 2, max_z + 3);
        for (block_x, block_y, block_z, block) in tree_blocks {
            if block_x < min_x || block_x > max_x || block_z < min_z || block_z > max_z {
                continue;
            }
            // 以实际世界状态为准：玩家砍掉的树干/树叶不能再被 LOD 生成出来。
            if world.block_at(block_x, block_y, block_z) != block {
                continue;
            }
            for face in FaceDirection::ALL {
                let neighbour = world.block_at(
                    block_x + face.offset[0],
                    block_y + face.offset[1],
                    block_z + face.offset[2],
                );
                if block::should_draw_face(block, neighbour) {
                    append_face(
                        &mut mesh,
                        BlockPosition {
                            x: block_x,
                            y: block_y,
                            z: block_z,
                            origin,
                        },
                        face,
                        block,
                        atlas,
                        world,
                    );
                }
            }
        }

        mesh
    }

    fn build_region(
        world: &mut GeneratedVoxelWorld,
        min_x: i64,
        max_x: i64,
        min_z: i64,
        max_z: i64,
        origin: DVec3,
        atlas: &Atlas,
    ) -> Self {
        // One-cell halo covers face visibility, corner AO and edge blending;
        // the much wider light halo lets the flood fill see enough terrain to
        // agree with neighbouring chunks across the border.
        let mut world = MeshingBlocks::new(
            world,
            min_x - LIGHT_HALO,
            max_x + LIGHT_HALO,
            min_z - LIGHT_HALO,
            max_z + LIGHT_HALO,
        );
        let mut mesh = Self::default();

        for z in min_z..=max_z {
            for x in min_x..=max_x {
                let max_y = world.top(x, z);
                for y in Y_MIN..=max_y {
                    let current = world.block_at(x, y, z);
                    if current == Block::Air {
                        continue;
                    }

                    for face in FaceDirection::ALL {
                        let neighbor = world.block_at(
                            x + face.offset[0],
                            y + face.offset[1],
                            z + face.offset[2],
                        );
                        if !block::should_draw_face(current, neighbor) {
                            continue;
                        }
                        append_face(
                            &mut mesh,
                            BlockPosition { x, y, z, origin },
                            face,
                            current,
                            atlas,
                            &mut world,
                        );
                    }
                }
            }
        }

        mesh
    }
}

/// A dense, column-major snapshot removes hash lookups from face/AO sampling.
struct MeshingBlocks {
    min_x: i64,
    min_z: i64,
    width: usize,
    depth: usize,
    blocks: Vec<Block>,
    tops: Vec<i64>,
    /// 洪泛天空光, 与 `blocks` 同布局 (`0..=LIGHT_MAX`).
    sky: Vec<u8>,
    /// 该快照里最高的列顶; 高于它的格子没有几何, 直接当作完全见天.
    light_ceiling: i64,
}

impl MeshingBlocks {
    const HEIGHT: usize = (Y_MAX - Y_MIN + 1) as usize;

    fn new(
        world: &mut GeneratedVoxelWorld,
        min_x: i64,
        max_x: i64,
        min_z: i64,
        max_z: i64,
    ) -> Self {
        let width = (max_x - min_x + 1) as usize;
        let depth = (max_z - min_z + 1) as usize;
        let mut result = Self {
            min_x,
            min_z,
            width,
            depth,
            blocks: vec![Block::Air; width * depth * Self::HEIGHT],
            tops: vec![Y_MIN; width * depth],
            sky: Vec::new(),
            light_ceiling: Y_MIN,
        };
        let columns = world.clone_columns_region(min_x, max_x, min_z, max_z);
        for column in columns {
            let index = result.column_index(column.x, column.z);
            let len = column.blocks.len().min(Self::HEIGHT);
            result.blocks[index * Self::HEIGHT..index * Self::HEIGHT + len]
                .copy_from_slice(&column.blocks[..len]);
            result.tops[index] = Y_MIN + len as i64 - 1;
        }
        for (&(x, y, z), &block) in world.edits() {
            if (min_x..=max_x).contains(&x) && (min_z..=max_z).contains(&z) {
                let index = result.column_index(x, z);
                result.blocks[index * Self::HEIGHT + (y - Y_MIN) as usize] = block;
                result.tops[index] = result.tops[index].max(y);
            }
        }

        let max_top = result
            .tops
            .iter()
            .copied()
            .max()
            .unwrap_or(Y_MIN)
            .clamp(Y_MIN, Y_MAX);
        let sky = light::flood_fill_sky(
            &result.blocks,
            width,
            depth,
            Self::HEIGHT,
            (max_top - Y_MIN) as usize,
        );
        result.sky = sky;
        result.light_ceiling = (max_top + 1).min(Y_MAX);
        result
    }

    fn column_index(&self, x: i64, z: i64) -> usize {
        (z - self.min_z) as usize * self.width + (x - self.min_x) as usize
    }

    fn top(&self, x: i64, z: i64) -> i64 {
        self.tops[self.column_index(x, z)]
    }
}

impl VoxelWorld for MeshingBlocks {
    fn block_at(&mut self, x: i64, y: i64, z: i64) -> Block {
        if !(Y_MIN..=Y_MAX).contains(&y) {
            return Block::Air;
        }
        debug_assert!(x >= self.min_x && x < self.min_x + self.width as i64);
        debug_assert!(z >= self.min_z && z < self.min_z + self.depth as i64);
        self.blocks[self.column_index(x, z) * Self::HEIGHT + (y - Y_MIN) as usize]
    }

    fn sky_light_at(&mut self, x: i64, y: i64, z: i64) -> u8 {
        if !(Y_MIN..=Y_MAX).contains(&y) || y > self.light_ceiling {
            return LIGHT_MAX;
        }
        debug_assert!(x >= self.min_x && x < self.min_x + self.width as i64);
        debug_assert!(z >= self.min_z && z < self.min_z + self.depth as i64);
        self.sky[self.column_index(x, z) * Self::HEIGHT + (y - Y_MIN) as usize]
    }
}

struct CachedChunkMesh {
    lod: LodLevel,
    mesh: VoxelMesh,
}

/// Retain geometry in chunk-local coordinates across whole-region requests.
#[derive(Default)]
struct RegionMeshCache {
    chunks: HashMap<ChunkKey, CachedChunkMesh>,
    edits: HashMap<(i64, i64, i64), Block>,
    rebuilt_chunks: usize,
}

impl RegionMeshCache {
    fn build(
        &mut self,
        world: &mut GeneratedVoxelWorld,
        center: (i64, i64),
        radius: i64,
        origin: DVec3,
        atlas: &Atlas,
        edits: &[BlockEdit],
    ) -> VoxelMesh {
        let next_edits: HashMap<_, _> = edits.iter().copied().collect();
        let changed: Vec<_> = self
            .edits
            .iter()
            .chain(next_edits.iter())
            .filter(|(position, _)| self.edits.get(position) != next_edits.get(position))
            .map(|(&(x, _, z), _)| (x, z))
            .collect();
        // An edit also changes the flood-filled light up to LIGHT_HALO blocks
        // away, so any cached chunk whose snapshot could see it must rebuild;
        // LOD_FAR_STEP separately covers the coarse neighbour samples.
        let invalidate = LIGHT_HALO.max(LOD_FAR_STEP);
        self.chunks.retain(|key, _| {
            !changed.iter().any(|&(x, z)| {
                x >= key.min_x() - invalidate
                    && x <= key.max_x() + invalidate
                    && z >= key.min_z() - invalidate
                    && z <= key.max_z() + invalidate
            })
        });
        self.edits = next_edits;
        world.replace_edits(edits.iter().map(|&((x, y, z), block)| (x, y, z, block)));
        self.rebuilt_chunks = 0;
        if radius <= CHUNK_SIZE {
            self.chunks.clear();
            return VoxelMesh::build_region(
                world,
                center.0 - radius,
                center.0 + radius,
                center.1 - radius,
                center.1 + radius,
                origin,
                atlas,
            );
        }
        let start = ChunkKey::from_block(center.0 - radius, center.1 - radius);
        let end = ChunkKey::from_block(center.0 + radius, center.1 + radius);
        self.chunks.retain(|key, _| {
            key.cx >= start.cx && key.cx <= end.cx && key.cz >= start.cz && key.cz <= end.cz
        });
        for cz in start.cz..=end.cz {
            for cx in start.cx..=end.cx {
                let key = ChunkKey { cx, cz };
                let lod = LodLevel::from_distance(key.distance_to(center.0, center.1));
                if self.chunks.get(&key).is_some_and(|entry| entry.lod == lod) {
                    continue;
                }
                let chunk_origin = DVec3::new(key.min_x() as f64, 0.0, key.min_z() as f64);
                let mesh = VoxelMesh::build_chunk(world, key, lod, chunk_origin, atlas, &[]);
                self.chunks.insert(key, CachedChunkMesh { lod, mesh });
                self.rebuilt_chunks += 1;
            }
        }
        let mut result = VoxelMesh::default();
        result
            .vertices
            .reserve(self.chunks.values().map(|e| e.mesh.vertices.len()).sum());
        result
            .indices
            .reserve(self.chunks.values().map(|e| e.mesh.indices.len()).sum());
        result.transparent_vertices.reserve(
            self.chunks
                .values()
                .map(|e| e.mesh.transparent_vertices.len())
                .sum(),
        );
        result.transparent_indices.reserve(
            self.chunks
                .values()
                .map(|e| e.mesh.transparent_indices.len())
                .sum(),
        );
        // Stable traversal preserves the existing transparent geometry ordering.
        for cz in start.cz..=end.cz {
            for cx in start.cx..=end.cx {
                let key = ChunkKey { cx, cz };
                let offset = DVec3::new(key.min_x() as f64, 0.0, key.min_z() as f64) - origin;
                append_cached_mesh(&mut result, &self.chunks[&key].mesh, offset);
            }
        }
        result
    }
}

fn append_cached_mesh(dst: &mut VoxelMesh, part: &VoxelMesh, offset: DVec3) {
    let translate = |vertex: &Vertex| {
        let mut vertex = *vertex;
        vertex.position = (DVec3::from_array(vertex.position.map(f64::from)) + offset)
            .as_vec3()
            .to_array();
        vertex
    };
    let base = dst.vertices.len() as u32;
    dst.vertices.extend(part.vertices.iter().map(translate));
    dst.indices
        .extend(part.indices.iter().map(|index| index + base));
    let base = dst.transparent_vertices.len() as u32;
    dst.transparent_vertices
        .extend(part.transparent_vertices.iter().map(translate));
    dst.transparent_indices
        .extend(part.transparent_indices.iter().map(|index| index + base));
}

#[cfg(test)]
fn append_mesh(dst: &mut VoxelMesh, part: VoxelMesh) {
    let base = dst.vertices.len() as u32;
    dst.vertices.extend(part.vertices);
    dst.indices
        .extend(part.indices.into_iter().map(|index| index + base));

    let transparent_base = dst.transparent_vertices.len() as u32;
    dst.transparent_vertices.extend(part.transparent_vertices);
    dst.transparent_indices.extend(
        part.transparent_indices
            .into_iter()
            .map(|index| index + transparent_base),
    );
}

#[cfg(test)]
fn lod_ring_cells(
    center: (i64, i64),
    radius: i64,
    inner_radius: i64,
    step: i64,
) -> Vec<(i64, i64, i64)> {
    debug_assert!(step > 0);
    let min_x = center.0 - radius;
    let max_x = center.0 + radius;
    let min_z = center.1 - radius;
    let max_z = center.1 + radius;
    let start_x = min_x.div_euclid(step) * step;
    let start_z = min_z.div_euclid(step) * step;
    let mut cells = Vec::new();
    let mut z = start_z;
    while z <= max_z {
        let mut x = start_x;
        while x <= max_x {
            let cell_max_x = x + step - 1;
            let cell_max_z = z + step - 1;
            let outside_inner = cell_max_x < center.0 - inner_radius
                || x > center.0 + inner_radius
                || cell_max_z < center.1 - inner_radius
                || z > center.1 + inner_radius;
            if outside_inner {
                cells.push((x, z, step));
            }
            x += step;
        }
        z += step;
    }
    cells
}

/// A representative column surface for a coarse LOD cell.
#[derive(Clone, Copy)]
struct LodSurface {
    /// Highest non-air block y in the representative column.
    y: i64,
    /// Absolute world y of the visible top surface (block top, or the fluid
    /// surface height).
    top: f32,
    block: Block,
}

fn lod_surface_from_block(y: i64, block: Block) -> LodSurface {
    let top = if block.is_fluid() {
        y as f32 + WATER_SURFACE_HEIGHT
    } else {
        (y + 1) as f32
    };
    LodSurface { y, top, block }
}

/// Whether a coarse LOD cell contains both fluid and non-fluid corners.
///
/// A single representative column cannot represent a shoreline correctly:
/// depending on which corner was sampled, the whole `step x step` quad either
/// covers water with land or covers land with water.  Such cells are rebuilt
/// at one-block resolution instead of being collapsed to one quad.
fn lod_cell_is_mixed(
    surfaces: &HashMap<(i64, i64), LodSurface>,
    x: i64,
    z: i64,
    step: i64,
) -> bool {
    let mut has_fluid = false;
    let mut has_land = false;
    for (dx, dz) in [(0, 0), (step, 0), (0, step), (step, step)] {
        if let Some(surface) = surfaces.get(&(x + dx, z + dz)) {
            if surface.block.is_fluid() {
                has_fluid = true;
            } else {
                has_land = true;
            }
        }
    }
    has_fluid && has_land
}

/// Absolute camera-relative bottom of a coarse LOD side face.
///
/// Fluids return the top itself: their side quads are both unnecessary (the
/// full-resolution mesher culls water/water and water/opaque faces) and
/// especially visible through the transparent surface as tall fins and
/// walls.  Opaque boundary cells keep the skirt to hide LOD cracks because
/// their top faces occlude it.
fn lod_side_bottom(
    surface: LodSurface,
    neighbour: Option<LodSurface>,
    force_skirt: bool,
    origin_y: f64,
    top: f32,
) -> f32 {
    if surface.block.is_fluid() {
        return top;
    }
    match neighbour {
        Some(neighbour) => {
            let neighbour_top = (neighbour.top as f64 - origin_y) as f32;
            if force_skirt {
                neighbour_top.min(top - LOD_SKIRT_HEIGHT)
            } else {
                neighbour_top
            }
        }
        None => top - LOD_SKIRT_HEIGHT,
    }
}

/// Build a mixed water/land coarse cell at one-block resolution.
///
/// Only cells on a shoreline are expanded this way.  Sampling the one-block
/// halo as well keeps the fine top and side quads connected to the coarse
/// neighbours instead of opening a crack between a fine sub-cell and the next
/// homogeneous LOD cell.
fn append_lod_fine_cell(
    mesh: &mut VoxelMesh,
    world: &mut GeneratedVoxelWorld,
    origin: DVec3,
    atlas: &Atlas,
    x: i64,
    z: i64,
    step: i64,
) {
    let sample_min_x = x - 1;
    let sample_max_x = x + step;
    let sample_min_z = z - 1;
    let sample_max_z = z + step;
    let mut surfaces: HashMap<(i64, i64), LodSurface> =
        HashMap::with_capacity(((step + 2).max(0) * (step + 2).max(0)) as usize);
    for sample_z in sample_min_z..=sample_max_z {
        for sample_x in sample_min_x..=sample_max_x {
            if let Some((y, block)) = world.terrain_surface_block(sample_x, sample_z) {
                surfaces.insert((sample_x, sample_z), lod_surface_from_block(y, block));
            }
        }
    }

    for cell_z in z..z + step {
        for cell_x in x..x + step {
            if let Some(&surface) = surfaces.get(&(cell_x, cell_z)) {
                append_lod_cell(
                    mesh, origin, atlas, cell_x, cell_z, 1, surface, &surfaces, false,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn append_lod_cell(
    mesh: &mut VoxelMesh,
    origin: DVec3,
    atlas: &Atlas,
    x: i64,
    z: i64,
    step: i64,
    surface: LodSurface,
    surfaces: &HashMap<(i64, i64), LodSurface>,
    force_skirt: bool,
) {
    let base = DVec3::new(x as f64, 0.0, z as f64) - origin;
    let bx = base.x as f32;
    let bz = base.z as f32;
    let top = (surface.top as f64 - origin.y) as f32;
    let step_f = step as f32;

    append_lod_quad(
        mesh,
        atlas,
        [
            [bx, top, bz],
            [bx + step_f, top, bz],
            [bx + step_f, top, bz + step_f],
            [bx, top, bz + step_f],
        ],
        [0.0, 1.0, 0.0],
        surface.block,
        Face::Top,
        x,
        surface.y,
        z,
    );

    let neighbor_bottom = |dx: i64, dz: i64| {
        lod_side_bottom(
            surface,
            surfaces.get(&(x + dx * step, z + dz * step)).copied(),
            force_skirt,
            origin.y,
            top,
        )
    };

    let sides = [
        (
            -1,
            0,
            [-1.0, 0.0, 0.0],
            [
                [bx, neighbor_bottom(-1, 0), bz],
                [bx, neighbor_bottom(-1, 0), bz + step_f],
                [bx, top, bz + step_f],
                [bx, top, bz],
            ],
        ),
        (
            1,
            0,
            [1.0, 0.0, 0.0],
            [
                [bx + step_f, neighbor_bottom(1, 0), bz + step_f],
                [bx + step_f, neighbor_bottom(1, 0), bz],
                [bx + step_f, top, bz],
                [bx + step_f, top, bz + step_f],
            ],
        ),
        (
            0,
            -1,
            [0.0, 0.0, -1.0],
            [
                [bx + step_f, neighbor_bottom(0, -1), bz],
                [bx, neighbor_bottom(0, -1), bz],
                [bx, top, bz],
                [bx + step_f, top, bz],
            ],
        ),
        (
            0,
            1,
            [0.0, 0.0, 1.0],
            [
                [bx, neighbor_bottom(0, 1), bz + step_f],
                [bx + step_f, neighbor_bottom(0, 1), bz + step_f],
                [bx + step_f, top, bz + step_f],
                [bx, top, bz + step_f],
            ],
        ),
    ];

    for (dx, dz, normal, positions) in sides {
        let bottom = neighbor_bottom(dx, dz);
        if bottom < top {
            append_lod_quad(
                mesh,
                atlas,
                positions,
                normal,
                surface.block,
                Face::Side,
                x,
                surface.y,
                z,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn append_lod_quad(
    mesh: &mut VoxelMesh,
    atlas: &Atlas,
    positions: [[f32; 3]; 4],
    normal: [f32; 3],
    block: Block,
    face: Face,
    x: i64,
    y: i64,
    z: i64,
) {
    let tile = atlas.tile_id(block, block.tile(face));
    let texture = face_texture(block, face, normal, x, y, z);
    let [region_x, region_y, columns, rows] = texture.region;
    let (u0, v0, u1, v1) = atlas.uv_region(tile, region_x, region_y, columns, rows);
    let tangent = normalize(subtract(positions[1], positions[0]));
    let bitangent = normalize(subtract(positions[0], positions[3]));
    let (vertices, indices) = if block.is_transparent() {
        (
            &mut mesh.transparent_vertices,
            &mut mesh.transparent_indices,
        )
    } else {
        (&mut mesh.vertices, &mut mesh.indices)
    };
    let base = vertices.len() as u32;
    let local_uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

    for (position, base_uv) in positions.into_iter().zip(local_uvs) {
        let local_uv = texture.local_uv(base_uv);
        let uv = [u0 + local_uv[0] * (u1 - u0), v0 + local_uv[1] * (v1 - v0)];
        vertices.push(Vertex {
            position,
            uv,
            normal,
            ao: 1.0,
            light: 1.0,
            material: block.render_material(),
            tangent,
            bitangent,
            local_uv,
            blend_tiles: [tile as u32; 4],
            texture_region: [region_x, region_y, columns, rows],
        });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn normalize(vector: [f32; 3]) -> [f32; 3] {
    let length = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    if length <= f32::EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [vector[0] / length, vector[1] / length, vector[2] / length]
    }
}

#[derive(Clone, Copy)]
struct FaceDirection {
    offset: [i64; 3],
    corners: [[f32; 3]; 4],
    normal: [f32; 3],
    /// 两条面内轴，用于从每个顶点角落采样 Corner AO 邻居。
    ao_axes: [[i64; 3]; 2],
    face: Face,
}

impl FaceDirection {
    const ALL: [Self; 6] = [
        Self {
            offset: [0, 1, 0],
            corners: [
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            normal: [0.0, 1.0, 0.0],
            ao_axes: [[1, 0, 0], [0, 0, 1]],
            face: Face::Top,
        },
        Self {
            offset: [0, -1, 0],
            corners: [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            normal: [0.0, -1.0, 0.0],
            ao_axes: [[1, 0, 0], [0, 0, 1]],
            face: Face::Bottom,
        },
        Self {
            offset: [0, 0, -1],
            corners: [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            normal: [0.0, 0.0, -1.0],
            ao_axes: [[1, 0, 0], [0, 1, 0]],
            face: Face::Side,
        },
        Self {
            offset: [0, 0, 1],
            corners: [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            normal: [0.0, 0.0, 1.0],
            ao_axes: [[1, 0, 0], [0, 1, 0]],
            face: Face::Side,
        },
        Self {
            offset: [-1, 0, 0],
            corners: [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            normal: [-1.0, 0.0, 0.0],
            ao_axes: [[0, 0, 1], [0, 1, 0]],
            face: Face::Side,
        },
        Self {
            offset: [1, 0, 0],
            corners: [
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            normal: [1.0, 0.0, 0.0],
            ao_axes: [[0, 0, 1], [0, 1, 0]],
            face: Face::Side,
        },
    ];
}

#[derive(Clone, Copy)]
struct BlockPosition {
    x: i64,
    y: i64,
    z: i64,
    origin: DVec3,
}

fn append_face(
    mesh: &mut VoxelMesh,
    position: BlockPosition,
    direction: FaceDirection,
    block: Block,
    atlas: &Atlas,
    world: &mut impl VoxelWorld,
) {
    let tile = atlas.tile_id(block, block.tile(direction.face));
    let texture = face_texture(
        block,
        direction.face,
        direction.normal,
        position.x,
        position.y,
        position.z,
    );
    let [region_x, region_y, columns, rows] = texture.region;
    let (u0, v0, u1, v1) = atlas.uv_region(tile, region_x, region_y, columns, rows);
    let (vertices, indices) = if block.is_transparent() {
        (
            &mut mesh.transparent_vertices,
            &mut mesh.transparent_indices,
        )
    } else {
        (&mut mesh.vertices, &mut mesh.indices)
    };
    let base = vertices.len() as u32;
    let block_origin =
        DVec3::new(position.x as f64, position.y as f64, position.z as f64) - position.origin;
    // 贴图被镜像时, 由 UV 定义的切线基也要跟着翻, 否则法线贴图的凹凸会朝反方向.
    let flipped = |axis: [f32; 3], mirror: bool| {
        if mirror {
            [-axis[0], -axis[1], -axis[2]]
        } else {
            axis
        }
    };
    let tangent = flipped(
        subtract(direction.corners[1], direction.corners[0]),
        texture.mirror_u,
    );
    let bitangent = flipped(
        subtract(direction.corners[0], direction.corners[3]),
        texture.mirror_v,
    );
    let mut blend_tiles = edge_blend_tiles(position, direction, block, tile, atlas, world);
    // 局部 UV 镜像后方块的两条边互换, 边缘混合的邻居也要跟着换.
    if texture.mirror_u {
        blend_tiles.swap(0, 1);
    }
    if texture.mirror_v {
        blend_tiles.swap(2, 3);
    }
    let local_uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

    for (corner, base_local_uv) in direction.corners.into_iter().zip(local_uvs) {
        let local_uv = texture.local_uv(base_local_uv);
        let uv = [u0 + local_uv[0] * (u1 - u0), v0 + local_uv[1] * (v1 - v0)];
        let corner = if block.is_fluid() && corner[1] > 0.5 {
            [corner[0], WATER_SURFACE_HEIGHT, corner[2]]
        } else {
            corner
        };
        let (ao, sky_light) = corner_ao_and_sky_light(world, position, direction, corner);
        vertices.push(Vertex {
            position: [
                (block_origin.x as f32) + corner[0],
                (block_origin.y as f32) + corner[1],
                (block_origin.z as f32) + corner[2],
            ],
            uv,
            normal: direction.normal,
            ao,
            light: sky_light,
            material: block.render_material(),
            tangent,
            bitangent,
            local_uv,
            blend_tiles,
            texture_region: [region_x, region_y, columns, rows],
        });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// 自然地表材质: 照片式的土/石/沙质地表, 也是唯一需要边缘混合的一类.
///
/// 这些方块用同一张照片连续铺开(见 [`face_texture`]), 因此材质交界处
/// 需要用邻居贴图做一点过渡; 同时它们也是唯一做逐单元镜像的一类.
fn blendable(block: Block) -> bool {
    matches!(
        block,
        Block::GrassBlock
            | Block::Dirt
            | Block::Stone
            | Block::Cobblestone
            | Block::Sand
            | Block::Sandstone
            | Block::Gravel
            | Block::Clay
            | Block::Bedrock
            | Block::Obsidian
    )
}

fn edge_blend_tiles(
    position: BlockPosition,
    direction: FaceDirection,
    block: Block,
    current_tile: usize,
    atlas: &Atlas,
    world: &mut impl VoxelWorld,
) -> [u32; 4] {
    let current_tile = current_tile as u32;
    if !blendable(block) {
        return [current_tile; 4];
    }

    let tangent = subtract(direction.corners[1], direction.corners[0]);
    let bitangent = subtract(direction.corners[0], direction.corners[3]);
    let to_offset = |axis: [f32; 3], sign: i64| {
        [
            axis[0].round() as i64 * sign,
            axis[1].round() as i64 * sign,
            axis[2].round() as i64 * sign,
        ]
    };
    let offsets = [
        to_offset(tangent, -1),
        to_offset(tangent, 1),
        to_offset(bitangent, -1),
        to_offset(bitangent, 1),
    ];
    offsets.map(|offset| {
        let x = position.x + offset[0];
        let y = position.y + offset[1];
        let z = position.z + offset[2];
        let neighbor = world.block_at(x, y, z);
        if !blendable(neighbor) {
            return current_tile;
        }
        let outer = world.block_at(
            x + direction.offset[0],
            y + direction.offset[1],
            z + direction.offset[2],
        );
        if !block::should_draw_face(neighbor, outer) {
            return current_tile;
        }
        atlas.tile_id(neighbor, neighbor.tile(direction.face)) as u32
    })
}

fn subtract(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// 一个方块面在 atlas 里选中的子区域, 以及该子区域是否需要 U/V 镜像.
///
/// `region` 的语义与 [`Vertex::texture_region`] 一致: `x/y` 是子区域下标,
/// `z/w` 是子区域网格的列数/行数.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceTexture {
    pub region: [u32; 4],
    pub mirror_u: bool,
    pub mirror_v: bool,
}

impl FaceTexture {
    /// 整张贴图铺满一个面 (玻璃、书架、熔炉、活塞、原木顶面这类有作者意图的面).
    const COMPLETE: Self = Self {
        region: [0, 0, 1, 1],
        mirror_u: false,
        mirror_v: false,
    };

    /// 面内局部坐标 `0..1` → 选中子区域内的局部坐标 (含镜像).
    pub fn local_uv(self, uv: [f32; 2]) -> [f32; 2] {
        [
            if self.mirror_u { 1.0 - uv[0] } else { uv[0] },
            if self.mirror_v { 1.0 - uv[1] } else { uv[1] },
        ]
    }
}

/// 方块面 → atlas 子区域 + 镜像标记. 网格生成器和诊断预览共用这一个实现,
/// 保证预览图与游戏里看到的一致.
///
/// 相邻方块按世界坐标连续取 2×2 子区域，不逐单元镜像：独立镜像会使
/// 共享边上另一坐标反向，破坏衔接。沙地的去重复由片元着色器连续混合完成。
/// 有作者意图的整张面仍然使用完整贴图，草侧面保持草皮朝上。
pub fn face_texture(
    block: Block,
    face: Face,
    normal: [f32; 3],
    x: i64,
    y: i64,
    z: i64,
) -> FaceTexture {
    let complete_face = matches!(
        block,
        Block::Glass | Block::Bookshelf | Block::Beehive | Block::Furnace | Block::Piston
    ) || (block == Block::OakLog && face != Face::Side);
    if complete_face {
        return FaceTexture::COMPLETE;
    }

    let (u_axis, v_axis) = face_axes(normal, x, y, z);
    // 草方块侧面必须草皮朝上, 所以只做左右镜像; 它横向取照片的一半,
    // 纵向铺满整张贴图, 好让草皮边缘永远落在方块顶部.
    let grass_side = block == Block::GrassBlock && face == Face::Side;
    let (mirror_u, mirror_v) = (false, false);

    let region_x = (u_axis.rem_euclid(2) ^ i64::from(mirror_u)) as u32;
    if grass_side {
        FaceTexture {
            region: [region_x, 0, 2, 1],
            mirror_u,
            mirror_v: false,
        }
    } else {
        let region_y = (v_axis.rem_euclid(2) ^ i64::from(mirror_v)) as u32;
        FaceTexture {
            region: [region_x, region_y, 2, 2],
            mirror_u,
            mirror_v,
        }
    }
}

/// 每个面的面内 U/V 轴在世界坐标里的取值 (返回方块坐标形式).
///
/// 让取样区域跟着每个面真实的 U/V 朝向走: 立方体相对的两个面绕序相反, 六个面
/// 都用同一组符号会在共享边界上取到互不相关的半边.
fn face_axes(normal: [f32; 3], x: i64, y: i64, z: i64) -> (i64, i64) {
    if normal[1] > 0.5 {
        (x, -z)
    } else if normal[1] < -0.5 {
        (x, z)
    } else if normal[2] < -0.5 {
        (-x, -y)
    } else if normal[2] > 0.5 {
        (x, -y)
    } else if normal[0] < -0.5 {
        (z, -y)
    } else {
        (-z, -y)
    }
}

/// 计算一个面角落的体素环境遮蔽。
///
/// 采样面外侧的两个侧邻居和一个对角邻居。两个侧邻居都被挡住时，
/// 对角方块不再额外增加遮蔽，避免狭窄角落被重复计算得过暗。
fn corner_ao_and_sky_light(
    world: &mut impl VoxelWorld,
    position: BlockPosition,
    direction: FaceDirection,
    corner: [f32; 3],
) -> (f32, f32) {
    let sign = |axis: [i64; 3]| {
        let component = if axis[0] != 0 {
            corner[0]
        } else if axis[1] != 0 {
            corner[1]
        } else {
            corner[2]
        };
        if component < 0.5 { -1 } else { 1 }
    };
    let a = direction.ao_axes[0];
    let b = direction.ao_axes[1];
    let sign_a = sign(a);
    let sign_b = sign(b);
    let mut sample = |offset: [i64; 3]| {
        world
            .block_at(
                position.x + offset[0],
                position.y + offset[1],
                position.z + offset[2],
            )
            .is_opaque()
    };
    let side_a = [
        direction.offset[0] + a[0] * sign_a,
        direction.offset[1] + a[1] * sign_a,
        direction.offset[2] + a[2] * sign_a,
    ];
    let side_b = [
        direction.offset[0] + b[0] * sign_b,
        direction.offset[1] + b[1] * sign_b,
        direction.offset[2] + b[2] * sign_b,
    ];
    let diagonal = [
        direction.offset[0] + a[0] * sign_a + b[0] * sign_b,
        direction.offset[1] + a[1] * sign_a + b[1] * sign_b,
        direction.offset[2] + a[2] * sign_a + b[2] * sign_b,
    ];
    let side_a_opaque = sample(side_a);
    let side_b_opaque = sample(side_b);
    let diagonal_opaque = sample(diagonal);
    // Move the closure out of scope so the light queries below can borrow the
    // world again; dropping a closure only ends its captured borrow.
    let _ = sample;
    let occlusion = if side_a_opaque && side_b_opaque {
        3
    } else {
        side_a_opaque as u8 + side_b_opaque as u8 + diagonal_opaque as u8
    };
    // 保留体素风格的明显接缝，但避免 AO 把底部压成死黑。
    let ao = [1.0, 0.84, 0.68, 0.52][occlusion as usize];

    // 天空光按同一个顶点平滑：取面外侧四格中非不透明格的平均；
    // 两侧都不透明时只信正前方，和 AO 的转角规则保持一致，
    // 避免已经压暗的角落再叠一层.
    let sample_light = |world: &mut dyn VoxelWorld, offset: [i64; 3]| {
        world.sky_light_at(
            position.x + offset[0],
            position.y + offset[1],
            position.z + offset[2],
        ) as u32
    };
    let mut sum = sample_light(world, direction.offset);
    let mut count = 1u32;
    if !(side_a_opaque && side_b_opaque) {
        if !side_a_opaque {
            sum += sample_light(world, side_a);
            count += 1;
        }
        if !side_b_opaque {
            sum += sample_light(world, side_b);
            count += 1;
        }
        if !diagonal_opaque {
            sum += sample_light(world, diagonal);
            count += 1;
        }
    }
    let sky_light = sum as f32 / (count * LIGHT_MAX as u32) as f32;
    (ao, sky_light)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::atlas;

    fn assert_same_mesh(a: &VoxelMesh, b: &VoxelMesh) {
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.transparent_indices, b.transparent_indices);
        // 光照只影响着色，不参与面剔除/AO/编辑；参考网格走的是默认
        // “完全见天”的 VoxelWorld，因此比较时把 light 归一化掉。
        let flatten = |vertices: &[Vertex]| -> Vec<Vertex> {
            vertices
                .iter()
                .map(|vertex| Vertex {
                    light: 0.0,
                    ..*vertex
                })
                .collect()
        };
        assert_eq!(
            bytemuck::cast_slice::<Vertex, u8>(&flatten(&a.vertices)),
            bytemuck::cast_slice::<Vertex, u8>(&flatten(&b.vertices))
        );
        assert_eq!(
            bytemuck::cast_slice::<Vertex, u8>(&flatten(&a.transparent_vertices)),
            bytemuck::cast_slice::<Vertex, u8>(&flatten(&b.transparent_vertices))
        );
    }

    #[test]
    fn dense_meshing_preserves_faces_ao_and_edits() {
        let atlas = atlas::build(std::path::Path::new("assets/textures")).expect("atlas");
        let mut world = GeneratedVoxelWorld::new(2026_0904);
        world.apply_edits([
            (-1, 110, -1, Block::Stone),
            (0, 110, -1, Block::Glass),
            (1, 110, 0, Block::Water),
            (0, 10, 0, Block::Air),
            (2, 127, 2, Block::Stone),
        ]);
        let mut reference = VoxelMesh::default();
        let origin = DVec3::new(-0.25, 60.5, 0.75);
        for z in -2..=2 {
            for x in -2..=2 {
                for y in Y_MIN..=Y_MAX {
                    let current = world.block_at(x, y, z);
                    if current == Block::Air {
                        continue;
                    }
                    for face in FaceDirection::ALL {
                        let neighbor = world.block_at(
                            x + face.offset[0],
                            y + face.offset[1],
                            z + face.offset[2],
                        );
                        if block::should_draw_face(current, neighbor) {
                            append_face(
                                &mut reference,
                                BlockPosition { x, y, z, origin },
                                face,
                                current,
                                &atlas,
                                &mut world,
                            );
                        }
                    }
                }
            }
        }
        let dense = VoxelMesh::build(&mut world, 0, 0, 2, origin, &atlas);
        assert_same_mesh(&reference, &dense);
    }

    #[test]
    fn meshed_vertices_bake_flood_filled_sky_light() {
        let atlas = atlas::build(std::path::Path::new("assets/textures")).expect("atlas");
        let mut world = GeneratedVoxelWorld::new(2026_0904);
        let ground = world.surface_top(0, 0);
        let roof_y = ground + 3;
        // A small roof directly above the origin shades the ground below while
        // the rest of the mesh stays open to the sky.
        world.apply_edits(
            (-2..=2)
                .flat_map(|x| (-2..=2).map(move |z| (x, roof_y, z, Block::Stone)))
                .collect::<Vec<_>>(),
        );
        let mesh = VoxelMesh::build(&mut world, 0, 0, 6, DVec3::ZERO, &atlas);
        let mut light: Vec<f32> = mesh
            .vertices
            .iter()
            .chain(&mesh.transparent_vertices)
            .map(|vertex| vertex.light)
            .collect();
        light.sort_by(f32::total_cmp);
        assert!(
            light.first().is_some_and(|&value| value < 0.99),
            "shaded ground under the roof must be baked darker"
        );
        assert!(
            light.last().is_some_and(|&value| value > 0.99),
            "open sky must stay fully lit"
        );
    }

    #[test]
    fn region_cache_reuses_rebases_and_invalidates_boundary_edits() {
        let atlas = atlas::build(std::path::Path::new("assets/textures")).expect("atlas");
        let seed = 2026_0904;
        let mut world = GeneratedVoxelWorld::new(seed);
        let mut cache = RegionMeshCache::default();
        let original = cache.build(&mut world, (16, 16), 33, DVec3::ZERO, &atlas, &[]);
        assert_eq!(cache.rebuilt_chunks, 9);
        let repeated = cache.build(&mut world, (16, 16), 33, DVec3::ZERO, &atlas, &[]);
        assert_eq!(cache.rebuilt_chunks, 0);
        assert_same_mesh(&original, &repeated);
        drop(original);
        drop(repeated);
        let origin = DVec3::new(17.25, 64.5, 16.75);
        let edits = [((32, 110, 32), Block::Glass)];
        let edited = cache.build(&mut world, (17, 16), 33, origin, &atlas, &edits);
        assert_eq!(cache.rebuilt_chunks, 4);
        let reference = VoxelMesh::build_parallel(MeshBuildInput {
            seed,
            center: (17, 16),
            radius: 33,
            origin,
            atlas: &atlas,
            initial_columns: &[],
            edits: &edits,
        });
        assert_same_mesh(&edited, &reference);
        let restored = cache.build(&mut world, (17, 16), 33, origin, &atlas, &[]);
        assert_eq!(cache.rebuilt_chunks, 4);
        let reference = VoxelMesh::build_parallel(MeshBuildInput {
            seed,
            center: (17, 16),
            radius: 33,
            origin,
            atlas: &atlas,
            initial_columns: &[],
            edits: &[],
        });
        assert_same_mesh(&restored, &reference);
        cache.build(&mut world, (1024, -1024), 33, origin, &atlas, &[]);
        assert_eq!(cache.chunks.len(), 16);
        assert_eq!(cache.rebuilt_chunks, 16);
    }

    #[test]
    #[ignore = "performance diagnostic: builds the full view and a movement update"]
    fn cached_view_mesh_benchmark() {
        let atlas = atlas::build(std::path::Path::new("assets/textures")).expect("atlas");
        let mut world = GeneratedVoxelWorld::new(2026_0904);
        let mut cache = RegionMeshCache::default();
        for center in [(0, 0), (0, 0), (8, 0), (32, 0)] {
            let start = std::time::Instant::now();
            let mesh = cache.build(&mut world, center, VIEW_RADIUS, DVec3::ZERO, &atlas, &[]);
            eprintln!(
                "center={center:?}, elapsed={:?}, rebuilt={}, cached={}, vertices={}",
                start.elapsed(),
                cache.rebuilt_chunks,
                cache.chunks.len(),
                mesh.vertices.len()
            );
            assert!(!mesh.is_empty());
        }
        let edits = [((128, 110, 0), Block::Glass), ((192, 110, 0), Block::Stone)];
        let cached = cache.build(
            &mut world,
            (32, 0),
            VIEW_RADIUS,
            DVec3::ZERO,
            &atlas,
            &edits,
        );
        let columns: Vec<_> = world.cached_columns().cloned().collect();
        let start = std::time::Instant::now();
        let reference = VoxelMesh::build_parallel(MeshBuildInput {
            seed: world.seed(),
            center: (32, 0),
            radius: VIEW_RADIUS,
            origin: DVec3::ZERO,
            atlas: &atlas,
            initial_columns: &columns,
            edits: &edits,
        });
        eprintln!(
            "uncached full rebuild with warm columns: {:?}",
            start.elapsed()
        );
        assert_same_mesh(&cached, &reference);
    }

    #[test]
    fn mesh_contains_only_exposed_faces() {
        let mut world = GeneratedVoxelWorld::new(2026_0904);
        let atlas = atlas::build(std::path::Path::new("assets/textures")).expect("atlas");
        let mesh = VoxelMesh::build(&mut world, 0, 0, 1, DVec3::ZERO, &atlas);

        assert!(!mesh.is_empty());
        assert_eq!(mesh.vertices.len() % 4, 0);
        assert_eq!(mesh.indices.len() % 6, 0);
        // 只要面剔除生效，就不会把完整地下体素的所有面送进去。
        assert!(mesh.vertices.len() < 3 * 3 * 3 * 128 * 6 * 4);
    }

    #[test]
    fn parallel_mesh_matches_serial_face_count() {
        let atlas = atlas::build(std::path::Path::new("assets/textures")).expect("atlas");
        let mut serial_world = GeneratedVoxelWorld::new(2026_0904);
        let serial = VoxelMesh::build(&mut serial_world, 0, 0, 2, DVec3::ZERO, &atlas);
        let parallel = VoxelMesh::build_parallel(MeshBuildInput {
            seed: 2026_0904,
            center: (0, 0),
            radius: 2,
            origin: DVec3::ZERO,
            atlas: &atlas,
            initial_columns: &[],
            edits: &[],
        });

        assert_eq!(parallel.vertices.len(), serial.vertices.len());
        assert_eq!(parallel.indices.len(), serial.indices.len());
        assert_eq!(
            parallel.transparent_vertices.len(),
            serial.transparent_vertices.len()
        );
        assert_eq!(
            parallel.transparent_indices.len(),
            serial.transparent_indices.len()
        );
    }

    const TOP: [f32; 3] = [0.0, 1.0, 0.0];

    /// 面内局部坐标 → 整张照片坐标系里的坐标 (0..columns / 0..rows).
    fn photo_uv(texture: FaceTexture, local: [f32; 2]) -> [f32; 2] {
        let local = texture.local_uv(local);
        [
            (texture.region[0] as f32 + local[0]).rem_euclid(texture.region[2] as f32),
            (texture.region[1] as f32 + local[1]).rem_euclid(texture.region[3] as f32),
        ]
    }

    #[test]
    fn neighbouring_natural_blocks_share_the_photo_edge() {
        // 单元内和单元之间都必须接着照片的同一条边, 否则地表会出现亮暗断层.
        // 用方块面边缘的"照片坐标"相等来验证: 左边块的右边缘 == 右边块的左边缘.
        for x in -8..8 {
            for z in -8..8 {
                for t in [0.0, 0.17, 0.63, 1.0] {
                    let left = face_texture(Block::Dirt, Face::Top, TOP, x, 0, z);
                    let right = face_texture(Block::Dirt, Face::Top, TOP, x + 1, 0, z);
                    let join_u = |a: FaceTexture, b: FaceTexture| {
                        let a = photo_uv(a, [1.0, t]);
                        let b = photo_uv(b, [0.0, t]);
                        (a[0] - b[0]).abs() + (a[1] - b[1]).abs()
                    };
                    assert!(join_u(left, right) < 1e-6, "x={x} z={z} 处照片横向断开");

                    let front = face_texture(Block::Dirt, Face::Top, TOP, x, 0, z);
                    let back = face_texture(Block::Dirt, Face::Top, TOP, x, 0, z + 1);
                    // 顶面的 V 轴是 -z, 所以 z 越大 V 越小: 两个方块共享的是
                    // 前者的低位边 (v=0) 和后者的高位边 (v=1).
                    let join_v = |a: FaceTexture, b: FaceTexture| {
                        let a = photo_uv(a, [t, 0.0]);
                        let b = photo_uv(b, [t, 1.0]);
                        (a[0] - b[0]).abs() + (a[1] - b[1]).abs()
                    };
                    assert!(join_v(front, back) < 1e-6, "x={x} z={z} 处照片纵向断开");
                }
            }
        }
    }

    #[test]
    fn one_photo_cell_shares_a_single_orientation() {
        // 同一个 2×2 单元里的四个方块必须共用一次镜像选择, 否则照片内部会断层.
        for cell in -4..4 {
            let mut seen: Option<(bool, bool)> = None;
            for u in 0..2 {
                for v in 0..2 {
                    // 顶面的面内轴是 (x, -z), 单元边界因此落在偶数 x / 奇数 z 上.
                    let x = cell * 2 + u;
                    let z = -cell * 2 - v;
                    let texture = face_texture(Block::Sand, Face::Top, TOP, x, 0, z);
                    let flips = (texture.mirror_u, texture.mirror_v);
                    match seen {
                        Some(expected) => assert_eq!(expected, flips, "单元 {cell} 内镜像不一致"),
                        None => seen = Some(flips),
                    }
                }
            }
        }
    }

    #[test]
    fn grass_sides_keep_the_turf_edge_on_top() {
        for x in -6..6 {
            for y in -3..3 {
                let texture =
                    face_texture(Block::GrassBlock, Face::Side, [-1.0, 0.0, 0.0], x, y, 0);
                assert!(!texture.mirror_v, "草方块侧面被上下翻转, 草皮会落到脚下");
                assert_eq!(texture.region[1], 0, "草方块侧面只取贴图的上一半");
                assert_eq!(texture.region[3], 1, "草方块侧面纵向铺满整张贴图");
            }
        }
    }

    #[test]
    fn authored_and_directional_faces_stay_unmirrored() {
        // 矿脉 / 木纹 / 木板行 / 水面有明确方向, 不做镜像.
        for block in [
            Block::CoalOre,
            Block::IronOre,
            Block::OakPlanks,
            Block::OakLeaves,
            Block::Water,
        ] {
            for x in -4..4 {
                for y in -4..4 {
                    let texture = face_texture(block, Face::Side, [-1.0, 0.0, 0.0], x, y, 0);
                    assert!(!texture.mirror_u && !texture.mirror_v, "{block:?} 不该镜像");
                }
            }
        }
        // 原木侧面依然是连续铺开的照片 (面内轴是 (z, -y), 所以取第 0 列第 1 行).
        let log = face_texture(Block::OakLog, Face::Side, [-1.0, 0.0, 0.0], 1, 1, 0);
        assert_eq!(log.region, [0, 1, 2, 2]);
        assert!(!log.mirror_u && !log.mirror_v);
    }

    #[test]
    fn complete_faces_still_use_the_whole_tile() {
        let texture = face_texture(Block::Bookshelf, Face::Side, [-1.0, 0.0, 0.0], 5, 3, 7);
        assert_eq!(texture, FaceTexture::COMPLETE);
        assert_eq!(texture.local_uv([0.25, 0.75]), [0.25, 0.75]);
    }

    #[test]
    fn coarse_grid_preload_only_generates_step_points() {
        let mut world = GeneratedVoxelWorld::new(2026_0904);
        world.preload_columns_grid_parallel(0, 0, 10, LOD_MID_STEP);
        // -10, -8, ..., 10 on both axes.
        assert_eq!(world.cached_column_count(), 11 * 11);
    }

    #[test]
    fn fluid_lod_sides_are_not_emitted() {
        let water = lod_surface_from_block(60, Block::Water);
        let lower_water = lod_surface_from_block(56, Block::Water);
        let land = lod_surface_from_block(60, Block::GrassBlock);
        let higher_land = lod_surface_from_block(64, Block::Stone);

        // Water must never receive an LOD side/skirt, even when a neighbour
        // is lower or the cell sits on a chunk boundary.
        assert_eq!(
            lod_side_bottom(water, Some(water), true, 0.0, water.top),
            water.top
        );
        assert_eq!(
            lod_side_bottom(water, Some(lower_water), false, 0.0, water.top),
            water.top
        );
        assert_eq!(
            lod_side_bottom(water, Some(lower_water), true, 0.0, water.top),
            water.top
        );
        assert_eq!(
            lod_side_bottom(water, None, false, 0.0, water.top),
            water.top
        );
        // Opaque boundary cells still get their crack-hiding skirt.
        assert_eq!(
            lod_side_bottom(land, Some(higher_land), true, 0.0, land.top),
            land.top - LOD_SKIRT_HEIGHT
        );
    }

    #[test]
    fn lod_cell_is_mixed_only_when_water_and_land_corners_are_present() {
        let water = lod_surface_from_block(60, Block::Water);
        let land = lod_surface_from_block(62, Block::GrassBlock);
        let mut surfaces = HashMap::new();
        for (x, z) in [(0, 0), (2, 0), (0, 2), (2, 2)] {
            surfaces.insert((x, z), water);
        }
        assert!(!lod_cell_is_mixed(&surfaces, 0, 0, 2));
        surfaces.insert((2, 2), land);
        assert!(lod_cell_is_mixed(&surfaces, 0, 0, 2));
        for (x, z) in [(0, 0), (2, 0), (0, 2), (2, 2)] {
            surfaces.insert((x, z), land);
        }
        assert!(!lod_cell_is_mixed(&surfaces, 0, 0, 2));
    }

    #[test]
    fn lod_ring_cells_skip_the_high_resolution_inner_square() {
        let mid = lod_ring_cells((0, 0), LOD_MID_RADIUS, LOD_NEAR_RADIUS, LOD_MID_STEP);
        assert!(!mid.is_empty());
        assert!(mid.iter().all(|&(x, z, step)| {
            !(x >= -LOD_NEAR_RADIUS
                && x + step - 1 <= LOD_NEAR_RADIUS
                && z >= -LOD_NEAR_RADIUS
                && z + step - 1 <= LOD_NEAR_RADIUS)
        }));

        let far = lod_ring_cells((0, 0), 192, LOD_MID_RADIUS, LOD_FAR_STEP);
        assert!(!far.is_empty());
        assert!(far.iter().all(|&(x, z, step)| {
            !(x >= -LOD_MID_RADIUS
                && x + step - 1 <= LOD_MID_RADIUS
                && z >= -LOD_MID_RADIUS
                && z + step - 1 <= LOD_MID_RADIUS)
        }));
    }

    #[test]
    fn coarse_lod_ring_builds_heightfield_faces() {
        let atlas = atlas::build(std::path::Path::new("assets/textures")).expect("atlas");
        let seed = 2026_0904;
        let mut world = GeneratedVoxelWorld::new(seed);
        // A small coarse grid is enough for the test annulus; the real worker
        // uses the same preload path with the full ring radii.
        world.preload_columns_grid_parallel(0, 0, 80, LOD_MID_STEP);
        let columns: Vec<Column> = world.cached_columns().cloned().collect();
        let mesh = VoxelMesh::build_lod_rings(MeshBuildInput {
            seed,
            center: (0, 0),
            radius: 80,
            origin: DVec3::ZERO,
            atlas: &atlas,
            initial_columns: &columns,
            edits: &[],
        });

        assert!(!mesh.is_empty());
        assert_eq!(mesh.vertices.len() % 4, 0);
        assert_eq!(mesh.indices.len() % 6, 0);
        assert_eq!(mesh.transparent_vertices.len() % 4, 0);
        assert_eq!(mesh.transparent_indices.len() % 6, 0);

        // The LOD ring must be substantially smaller than the full-resolution
        // 161x161-cell grid it replaces at the same radius.
        let cells = lod_ring_cells((0, 0), 80, LOD_NEAR_RADIUS, LOD_MID_STEP);
        assert!(cells.len() < (161 * 161) / 3);
    }

    #[test]
    #[ignore = "heavy: builds the real 192-block LOD mesh"]
    fn full_view_lod_mesh_builds() {
        let atlas = atlas::build(std::path::Path::new("assets/textures")).expect("atlas");
        let seed = 2026_0904;
        let view_radius = 192;
        let mut world = GeneratedVoxelWorld::new(seed);
        world.preload_columns_parallel(
            -LOD_NEAR_RADIUS - 1,
            LOD_NEAR_RADIUS + 1,
            -LOD_NEAR_RADIUS - 1,
            LOD_NEAR_RADIUS + 1,
        );
        world.preload_columns_grid_parallel(0, 0, LOD_MID_RADIUS, LOD_MID_STEP);
        world.preload_columns_grid_parallel(0, 0, view_radius, LOD_FAR_STEP);
        let columns: Vec<Column> = world.cached_columns().cloned().collect();

        let start = std::time::Instant::now();
        let mesh = VoxelMesh::build_parallel(MeshBuildInput {
            seed,
            center: (0, 0),
            radius: view_radius,
            origin: DVec3::ZERO,
            atlas: &atlas,
            initial_columns: &columns,
            edits: &[],
        });
        eprintln!(
            "view LOD built in {:?}: {} vertices, {} indices",
            start.elapsed(),
            mesh.vertices.len(),
            mesh.indices.len()
        );
        assert!(!mesh.is_empty());
    }
}
