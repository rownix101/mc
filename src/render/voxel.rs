//! CPU 体素面剔除与网格生成。
//!
//! 当前先按玩家附近的方块生成 camera-relative 网格。网格只包含暴露面，
//! 因此地下的大量方块不会进入 GPU；后续可把这个接口替换成 chunk mesher。

use std::collections::HashMap;
use std::sync::{Arc, mpsc};
use std::thread;

use bytemuck::{Pod, Zeroable};
use glam::DVec3;
use rayon::prelude::*;

use crate::world::atlas::Atlas;
use crate::world::block::{self, Block, Face};
use crate::world::column::{Column, Y_MAX, Y_MIN};
use crate::world::voxel::{GeneratedVoxelWorld, VoxelWorld};

/// Minecraft-style fluid surface height. Water occupies a voxel for world
/// storage and placement, but its visible surface sits slightly below the
/// voxel boundary instead of looking like a full solid cube.
const WATER_SURFACE_HEIGHT: f32 = 0.875;

/// GPU 顶点：位置、atlas UV、面法线、环境遮蔽值，以及透明材质标记。
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
    /// 1 = 流体，0 = 其它材质。透明网格中的玻璃仍保持 0。
    pub material: f32,
}

impl Vertex {
    pub const fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x2,
            2 => Float32x3,
            3 => Float32,
            4 => Float32
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

struct MeshRequest {
    request_id: u64,
    center: (i64, i64),
    radius: i64,
    origin: DVec3,
    edits: Vec<BlockEdit>,
}

pub struct MeshResult {
    pub request_id: u64,
    pub center: (i64, i64),
    pub origin: DVec3,
    pub mesh: VoxelMesh,
}

struct MeshBuildInput<'a> {
    seed: u64,
    center: (i64, i64),
    radius: i64,
    origin: DVec3,
    atlas: &'a Atlas,
    initial_columns: &'a [Column],
    edits: &'a [BlockEdit],
}

/// 后台生成体素网格，避免移动时在渲染线程同步扫描整片区域。
///
/// 生成器线程持有自己的世界缓存。地形是确定性的，因此它和主线程的
/// 物理世界可以独立查询；主线程只接收完成的网格并上传 GPU buffer。
pub struct VoxelMeshWorker {
    requests: mpsc::Sender<MeshRequest>,
    results: mpsc::Receiver<MeshResult>,
    next_request_id: u64,
    latest_request_id: Option<u64>,
}

impl VoxelMeshWorker {
    pub fn new(seed: u64, atlas: Arc<Atlas>, initial_columns: Vec<Column>) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<MeshRequest>();
        let (result_tx, result_rx) = mpsc::channel::<MeshResult>();

        thread::spawn(move || {
            let mut world = GeneratedVoxelWorld::with_columns(seed, initial_columns);
            while let Ok(mut request) = request_rx.recv() {
                // 如果玩家在一次网格生成期间移动了很远，只处理队列中最新的
                // 请求，避免把已经过时的整片网格继续排队生成。
                while let Ok(newer) = request_rx.try_recv() {
                    request = newer;
                }

                world.preload_columns_parallel(
                    request.center.0 - request.radius,
                    request.center.0 + request.radius,
                    request.center.1 - request.radius,
                    request.center.1 + request.radius,
                );
                let cached_columns: Vec<Column> = world.cached_columns().cloned().collect();
                let mesh = VoxelMesh::build_parallel(MeshBuildInput {
                    seed,
                    center: request.center,
                    radius: request.radius,
                    origin: request.origin,
                    atlas: &atlas,
                    initial_columns: &cached_columns,
                    edits: &request.edits,
                });
                if result_tx
                    .send(MeshResult {
                        request_id: request.request_id,
                        center: request.center,
                        origin: request.origin,
                        mesh,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });

        Self {
            requests: request_tx,
            results: result_rx,
            next_request_id: 0,
            latest_request_id: None,
        }
    }

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
                request_id,
                center,
                radius,
                origin,
                edits,
            })
            .is_ok()
        {
            self.latest_request_id = Some(request_id);
        }
    }

    /// 取出当前最新请求的结果；已经过时的结果直接丢弃。
    pub fn try_take_latest(&mut self) -> Option<MeshResult> {
        let expected_id = self.latest_request_id?;
        let mut latest = None;
        while let Ok(result) = self.results.try_recv() {
            if result.request_id == expected_id {
                latest = Some(result);
            }
        }
        if latest.is_some() {
            self.latest_request_id = None;
        }
        latest
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

    /// Build independent x/z regions in parallel. Each region owns a local
    /// deterministic world view, so no lock is needed around column generation
    /// or mesh writes. The one-column border in the preload filter covers
    /// neighbor and AO samples at region edges.
    fn build_parallel(input: MeshBuildInput<'_>) -> Self {
        const TILE_SIZE: i64 = 8;
        let min_x = input.center.0 - input.radius;
        let max_x = input.center.0 + input.radius;
        let min_z = input.center.1 - input.radius;
        let max_z = input.center.1 + input.radius;

        // The worker cache is already keyed by column coordinates. Index it
        // once here instead of scanning the complete cache for every tile.
        let column_index: HashMap<(i64, i64), &Column> = input
            .initial_columns
            .iter()
            .map(|column| ((column.x, column.z), column))
            .collect();

        // Edits are copied into a column index once. Each tile only needs its
        // own edits plus a one-column border for face/AO neighbor samples.
        let mut edits_by_column: HashMap<(i64, i64), Vec<BlockEdit>> = HashMap::new();
        for &(position, block) in input.edits {
            edits_by_column
                .entry((position.0, position.2))
                .or_default()
                .push((position, block));
        }

        let mut regions = Vec::new();
        let mut region_min_x = min_x;
        while region_min_x <= max_x {
            let region_max_x = (region_min_x + TILE_SIZE - 1).min(max_x);
            let mut region_min_z = min_z;
            while region_min_z <= max_z {
                let region_max_z = (region_min_z + TILE_SIZE - 1).min(max_z);
                regions.push((region_min_x, region_max_x, region_min_z, region_max_z));
                region_min_z = region_max_z + 1;
            }
            region_min_x = region_max_x + 1;
        }

        let parts: Vec<Self> = regions
            .into_par_iter()
            .map(|(region_min_x, region_max_x, region_min_z, region_max_z)| {
                let region_columns: Vec<Column> = (region_min_z - 1..=region_max_z + 1)
                    .flat_map(|z| (region_min_x - 1..=region_max_x + 1).map(move |x| (x, z)))
                    .filter_map(|coordinate| column_index.get(&coordinate).cloned())
                    .cloned()
                    .collect();
                let mut world = GeneratedVoxelWorld::with_columns(input.seed, region_columns);
                for z in region_min_z - 1..=region_max_z + 1 {
                    for x in region_min_x - 1..=region_max_x + 1 {
                        if let Some(edits) = edits_by_column.get(&(x, z)) {
                            for &((edit_x, edit_y, edit_z), block) in edits {
                                world.set_block(edit_x, edit_y, edit_z, block);
                            }
                        }
                    }
                }
                Self::build_region(
                    &mut world,
                    region_min_x,
                    region_max_x,
                    region_min_z,
                    region_max_z,
                    input.origin,
                    input.atlas,
                )
            })
            .collect();

        let mut mesh = Self::default();
        for part in parts {
            let base = mesh.vertices.len() as u32;
            mesh.vertices.extend(part.vertices);
            mesh.indices
                .extend(part.indices.into_iter().map(|index| index + base));

            let transparent_base = mesh.transparent_vertices.len() as u32;
            mesh.transparent_vertices.extend(part.transparent_vertices);
            mesh.transparent_indices.extend(
                part.transparent_indices
                    .into_iter()
                    .map(|index| index + transparent_base),
            );
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
        let mut mesh = Self::default();

        for z in min_z..=max_z {
            for x in min_x..=max_x {
                // 没有必要扫描到世界上限：列生成器在地形顶端以上只有空气，
                // 海洋列最多还会存水到 SEA_Y。
                let max_y = world.surface_top(x, z).clamp(Y_MIN, Y_MAX);
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
                            world,
                        );
                    }
                }
            }
        }

        mesh
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
    world: &mut GeneratedVoxelWorld,
) {
    let tile = atlas.tile_id(block, block.tile(direction.face));
    let (u0, v0, u1, v1) = atlas.uv(tile);
    let (vertices, indices) = if block.is_transparent() {
        (
            &mut mesh.transparent_vertices,
            &mut mesh.transparent_indices,
        )
    } else {
        (&mut mesh.vertices, &mut mesh.indices)
    };
    let base = vertices.len() as u32;
    let uvs = [[u0, v1], [u1, v1], [u1, v0], [u0, v0]];
    let block_origin =
        DVec3::new(position.x as f64, position.y as f64, position.z as f64) - position.origin;

    for (corner, uv) in direction.corners.into_iter().zip(uvs) {
        let corner = if block.is_fluid() && corner[1] > 0.5 {
            [corner[0], WATER_SURFACE_HEIGHT, corner[2]]
        } else {
            corner
        };
        let ao = corner_ao(world, position, direction, corner);
        vertices.push(Vertex {
            position: [
                (block_origin.x as f32) + corner[0],
                (block_origin.y as f32) + corner[1],
                (block_origin.z as f32) + corner[2],
            ],
            uv,
            normal: direction.normal,
            ao,
            material: block.is_fluid() as u8 as f32,
        });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// 计算一个面角落的体素环境遮蔽。
///
/// 采样面外侧的两个侧邻居和一个对角邻居。两个侧邻居都被挡住时，
/// 对角方块不再额外增加遮蔽，避免狭窄角落被重复计算得过暗。
fn corner_ao(
    world: &mut GeneratedVoxelWorld,
    position: BlockPosition,
    direction: FaceDirection,
    corner: [f32; 3],
) -> f32 {
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
    let side_a = sample(side_a);
    let side_b = sample(side_b);
    let diagonal = sample(diagonal);
    let occlusion = if side_a && side_b {
        3
    } else {
        side_a as u8 + side_b as u8 + diagonal as u8
    };
    // 保留体素风格的明显接缝，但避免 AO 把底部压成死黑。
    [1.0, 0.84, 0.68, 0.52][occlusion as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::atlas;

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
}
