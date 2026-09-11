//! 世界空间辐照度探针体积 (GI 阶段 B 第一步).
//!
//! 以固定间距的 3D 探针网格覆盖相机附近区域, 每个探针用 SH L1 记录
//! "天空可见度随方向分布" 的低阶近似. 片元着色器按法线重建可见度, 得到
//! 带方向性的环境光: 山坳 / 洞口 / 树荫从上方收到的天空光明显少于平地.
//!
//! 只烘焙天空可见度 (与昼夜无关), 天空颜色仍在着色器里按时间计算; 首个
//! 命中方块的颜色 (颜色反弹) 留待后续阶段.
//!
//! 探针在网格 worker 线程上用确定性世界生成烘焙, 因此能看见地形 / 树冠 /
//! 水体和玩家编辑, 不需要额外维护一份 GPU 占用表示.

use std::f32::consts::PI;

use glam::{DVec3, Vec3};

use crate::world::block::Block;
use crate::world::column::{Y_MAX, Y_MIN};
use crate::world::voxel::{GeneratedVoxelWorld, VoxelWorld};

/// 探针间距 (格). 24x16x24 个探针覆盖 192x128x192 格.
pub const SPACING: i64 = 8;
pub const DIM_X: usize = 24;
pub const DIM_Y: usize = 16;
pub const DIM_Z: usize = 24;
/// 探针总数, 也就是 3D 纹理的 texel 数.
pub const TEXELS: usize = DIM_X * DIM_Y * DIM_Z;
/// 单条可见性射线的最大步数; 命中距离也用它归一化成部分可见.
pub const RAY_RANGE: i64 = 40;
/// 每个探针投射的方向数. 只投 6 个轴向无法表达"屋顶遮住整片上方",
/// 16 个 Fibonacci 均匀球面方向足以喂饱 SH L1.
const DIRECTIONS: usize = 16;

/// SH L1 基函数 (Y00, Y1-1*y, Y10*z, Y11*x).
const SH_Y00: f32 = 0.282_094_8;
const SH_Y1: f32 = 0.488_602_5;

/// 一帧相机附近的探针网格.
#[derive(Clone, Debug)]
pub struct ProbeVolume {
    /// 世界坐标下网格最小角 (texel 0 的下边界).
    pub origin: DVec3,
    /// 每个探针 4 个 SH L1 系数, 顺序为 (z*DIM_Y + y)*DIM_X + x.
    pub texels: Vec<[f32; 4]>,
}

impl ProbeVolume {
    /// 以 center 为中心烘焙一份探针.
    pub fn bake(world: &mut GeneratedVoxelWorld, center: (i64, i64)) -> Self {
        Self::bake_in(world, center)
    }

    /// 与 bake 相同的算法, 但接受任意 VoxelWorld, 便于测试.
    pub fn bake_in(world: &mut impl VoxelWorld, center: (i64, i64)) -> Self {
        let origin = DVec3::new(
            (center.0 - (DIM_X as i64 * SPACING) / 2) as f64,
            Y_MIN as f64,
            (center.1 - (DIM_Z as i64 * SPACING) / 2) as f64,
        );
        let directions = sphere_directions();
        let mut texels = Vec::with_capacity(TEXELS);
        for z in 0..DIM_Z {
            for y in 0..DIM_Y {
                for x in 0..DIM_X {
                    let px = origin.x as i64 + x as i64 * SPACING + SPACING / 2;
                    let py = Y_MIN + y as i64 * SPACING + SPACING / 2;
                    let pz = origin.z as i64 + z as i64 * SPACING + SPACING / 2;
                    texels.push(sky_visibility(world, px, py, pz, &directions));
                }
            }
        }
        Self { origin, texels }
    }

    /// texel 索引, 与纹理写入顺序一致.
    pub fn index(x: usize, y: usize, z: usize) -> usize {
        (z * DIM_Y + y) * DIM_X + x
    }
}

/// 按法线从 SH L1 系数重建天空可见度, 并夹到 0..=1.
///
/// 与 WGSL 里的展开保持一致; 单元测试用它验证烘焙结果.
pub fn reconstruct(sh: [f32; 4], normal: Vec3) -> f32 {
    let value = sh[0] * SH_Y00
        + sh[1] * SH_Y1 * normal.y
        + sh[2] * SH_Y1 * normal.z
        + sh[3] * SH_Y1 * normal.x;
    value.clamp(0.0, 1.0)
}

/// Fibonacci 球面上的固定方向集合 (确定性, 与方向顺序无关).
fn sphere_directions() -> Vec<Vec3> {
    let golden = PI * (3.0 - 5.0f32.sqrt());
    (0..DIRECTIONS)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / DIRECTIONS as f32;
            let radius = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden * i as f32;
            Vec3::new(theta.cos() * radius, y, theta.sin() * radius)
        })
        .collect()
}

/// 在 (x, y, z) 处沿所有采样方向求天空可见度并投影到 SH L1.
fn sky_visibility(
    world: &mut impl VoxelWorld,
    x: i64,
    y: i64,
    z: i64,
    directions: &[Vec3],
) -> [f32; 4] {
    if blocks_sky(world.block_at(x, y, z)) {
        return [0.0; 4];
    }
    let weight = 4.0 * PI / directions.len() as f32;
    let mut sh = [0.0f32; 4];
    for direction in directions {
        let visibility = ray_visibility(world, x, y, z, *direction);
        sh[0] += visibility * SH_Y00 * weight;
        sh[1] += visibility * SH_Y1 * direction.y * weight;
        sh[2] += visibility * SH_Y1 * direction.z * weight;
        sh[3] += visibility * SH_Y1 * direction.x * weight;
    }
    sh
}

/// 沿单位方向前进, 返回 0..=1 的天空可见度: 命中越近越暗, 逃逸到世界顶为 1.
fn ray_visibility(world: &mut impl VoxelWorld, x: i64, y: i64, z: i64, direction: Vec3) -> f32 {
    for step in 1..=RAY_RANGE {
        let distance = step as f32;
        let sy = y + (direction.y * distance).round() as i64;
        if sy > Y_MAX {
            return 1.0;
        }
        if sy < Y_MIN {
            return 0.0;
        }
        let sx = x + (direction.x * distance).round() as i64;
        let sz = z + (direction.z * distance).round() as i64;
        if blocks_sky(world.block_at(sx, sy, sz)) {
            return (distance / RAY_RANGE as f32).clamp(0.0, 1.0);
        }
    }
    1.0
}

/// 探针射线是否被挡住: 不透明 / 水 / 树叶挡光, 玻璃不挡.
fn blocks_sky(block: Block) -> bool {
    block.is_opaque() || block.is_fluid() || block == Block::OakLeaves
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 下方到 ground 为止是实心, 其余为空; 可选一层铺满的水平屋顶.
    struct TestWorld {
        ground: i64,
        roof: Option<i64>,
    }

    impl VoxelWorld for TestWorld {
        fn block_at(&mut self, _x: i64, y: i64, _z: i64) -> Block {
            if self.roof == Some(y) || y < self.ground {
                Block::Stone
            } else {
                Block::Air
            }
        }
    }

    fn visibility_at(world: &mut impl VoxelWorld, y: i64) -> [f32; 4] {
        let directions = sphere_directions();
        sky_visibility(world, 0, y, 0, &directions)
    }

    #[test]
    fn open_sky_reconstructs_to_full_visibility() {
        let mut world = TestWorld {
            ground: Y_MIN,
            roof: None,
        };
        let sh = visibility_at(&mut world, 64);
        assert!((reconstruct(sh, Vec3::Y) - 1.0).abs() < 0.001);
        assert!((reconstruct(sh, Vec3::NEG_Y) - 1.0).abs() < 0.001);
    }

    #[test]
    fn ground_blocks_downward_visibility() {
        let mut world = TestWorld {
            ground: 64,
            roof: None,
        };
        let sh = visibility_at(&mut world, 65);
        assert!(
            reconstruct(sh, Vec3::Y) > 0.9,
            "open sky above must stay bright"
        );
        assert!(
            reconstruct(sh, Vec3::NEG_Y) < 0.2,
            "ground right below must be dark"
        );
    }

    #[test]
    fn roof_shadows_the_probe_below() {
        let mut open = TestWorld {
            ground: 64,
            roof: None,
        };
        let mut roofed = TestWorld {
            ground: 64,
            roof: Some(70),
        };
        let open_sh = visibility_at(&mut open, 69);
        let roofed_sh = visibility_at(&mut roofed, 69);
        let open_up = reconstruct(open_sh, Vec3::Y);
        let roofed_up = reconstruct(roofed_sh, Vec3::Y);
        assert!(
            roofed_up < open_up - 0.2,
            "a ceiling overhead must cut the sky: {roofed_up} vs {open_up}"
        );
        assert!(roofed_up < 0.8);
    }

    #[test]
    fn probe_inside_solid_is_dark() {
        let mut world = TestWorld {
            ground: 64,
            roof: None,
        };
        let sh = visibility_at(&mut world, 10);
        assert_eq!(sh, [0.0; 4]);
    }

    #[test]
    fn bake_fills_grid_and_darkens_buried_probes() {
        let mut world = TestWorld {
            ground: 64,
            roof: None,
        };
        let volume = ProbeVolume::bake_in(&mut world, (0, 0));
        assert_eq!(volume.texels.len(), TEXELS);
        assert_eq!(volume.origin, DVec3::new(-96.0, Y_MIN as f64, -96.0));
        // y index 4 -> probe center y = 36, buried.
        let buried = volume.texels[ProbeVolume::index(12, 4, 12)];
        assert_eq!(buried, [0.0; 4]);
        // y index 15 -> probe center y = 124, open sky.
        let open = volume.texels[ProbeVolume::index(12, 15, 12)];
        assert!(reconstruct(open, Vec3::Y) > 0.9);
    }
}
