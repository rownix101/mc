//! 树木装饰器: 在草地表面概率性放置橡树.
//!
//! 与 `ColumnGen` 正交: `ColumnGen` 产出纯地形柱, `TreeDecorator` 扫描一个区域、
//! 挑候选位置、输出 `Vec<Tree>`. 调用方把 `Tree::blocks()` 写入已有柱即可.
//! 不引入新的 Chunk 结构.
//!
//! 橡树形状 (经典 Minecraft 风格):
//!
//! ```text
//!        L L L
//!      L L L L L        ← 顶层 (y+1): 十字 3x3
//!      L L L L L        ← 中层 (y+0): 5x5 切四角
//!        L L L          ← 底层 (y-1): 5x5 切四角 + 十字
//!          T            ← 树干
//!          T
//!          T
//!          T
//!        G G G          ← 草地表面
//! ```
//!
//! - 树干高 4--5 格, 由位置 hash 决定.
//! - 树冠在树干顶部上下分布, 半径 2, 高度 3.
//! - 放置条件: 草方块表面 (高度 `SEA_LEVEL < h <= SEA_LEVEL + 10`, 非沙滩/水/高山).
//! - 最小间距: 任意两棵树 trunk 水平距离 >= 3 (防树冠重叠).
//! - 概率: `density` 默认 0.04 (~4% 的草地格).

use super::block::Block;
use super::continent::{SEA_LEVEL, WorldHeightmap};
use super::plains::splitmix64;

/// 一棵树: 位置 + 形状参数.
#[derive(Clone, Debug)]
pub struct Tree {
    pub x: i64,
    pub z: i64,
    /// 地面 y (草方块顶部).
    pub ground: i64,
    /// 树干高度 (格).
    pub height: i64,
}

impl Tree {
    /// 该树占用的全部方块, 以 `(x, y, z, Block)` 产出.
    pub fn blocks(&self) -> Vec<(i64, i64, i64, Block)> {
        let mut out = Vec::new();
        let top = self.ground + 1;

        // 树干.
        for y in top..top + self.height {
            out.push((self.x, y, self.z, Block::OakLog));
        }

        let canopy_center = top + self.height - 1; // 树冠中心 y
        for dy in -1..=1 {
            let y = canopy_center + dy;
            for dx in -2..=2i64 {
                for dz in -2..=2i64 {
                    // 切四角.
                    if dx.abs() == 2 && dz.abs() == 2 {
                        continue;
                    }
                    let dist = dx.abs() + dz.abs();
                    // 顶层 (dy=+1): 只留十字 (dist <= 1).
                    if dy == 1 && dist > 1 {
                        continue;
                    }
                    // 底层 (dy=-1): 只留十字 (dist <= 1), 避免地面过宽.
                    if dy == -1 && dist > 1 {
                        continue;
                    }
                    // 中心列 (dx==0 && dz==0) 已被树干占据, 跳过.
                    if dx == 0 && dz == 0 {
                        continue;
                    }
                    out.push((self.x + dx, y, self.z + dz, Block::OakLeaves));
                }
            }
        }

        out
    }
}

/// 区域树木装饰器.
pub struct TreeDecorator {
    pub seed: u64,
    /// 每个候选格被种上树的概率.
    pub density: f64,
}

impl TreeDecorator {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            density: 0.04,
        }
    }

    pub fn with_density(seed: u64, density: f64) -> Self {
        Self { seed, density }
    }

    /// 在 `[x0, x1) × [z0, z1)` 范围内生成树.
    ///
    /// `height_at(x, z) -> f64` 返回表面高度 (格). 树只种在草地:
    /// `SEA_LEVEL < h <= SEA_LEVEL + 10.0`.
    pub fn generate(
        &self,
        world: &WorldHeightmap,
        x0: i64,
        x1: i64,
        z0: i64,
        z1: i64,
    ) -> Vec<Tree> {
        let w = (x1 - x0) as usize;
        let h = (z1 - z0) as usize;
        let mut candidates = vec![false; w * h];
        let mut ground_y = vec![0i64; w * h];

        // 第一遍: 标记候选 + 记录地面高.
        for zi in 0..h {
            for xi in 0..w {
                let x = x0 + xi as i64;
                let z = z0 + zi as i64;
                let hi = world.height(x, z);
                let ground = hi.floor() as i64;
                ground_y[zi * w + xi] = ground;
                if is_grass_surface(hi) && self.roll(x, z) {
                    candidates[zi * w + xi] = true;
                }
            }
        }

        // 第二遍: spacing 检查. 全方向 5×5 邻域, 若已有保留候选且距离 < 3, 则跳过.
        let mut keep = vec![false; w * h];
        for zi in 0..h {
            for xi in 0..w {
                if !candidates[zi * w + xi] {
                    continue;
                }
                let mut conflict = false;
                for dzi in -3i64..=3 {
                    for dxi in -3i64..=3 {
                        if dxi == 0 && dzi == 0 {
                            continue;
                        }
                        let nzi = zi as i64 + dzi;
                        let nxi = xi as i64 + dxi;
                        if nzi < 0 || nxi < 0 || nzi >= h as i64 || nxi >= w as i64 {
                            continue;
                        }
                        if keep[nzi as usize * w + nxi as usize] {
                            let dist2 = dxi * dxi + dzi * dzi;
                            if dist2 < 9 {
                                conflict = true;
                                break;
                            }
                        }
                    }
                    if conflict {
                        break;
                    }
                }
                if !conflict {
                    keep[zi * w + xi] = true;
                }
            }
        }

        // 第三遍: 构造 Tree.
        let mut trees = Vec::new();
        for zi in 0..h {
            for xi in 0..w {
                if !keep[zi * w + xi] {
                    continue;
                }
                let x = x0 + xi as i64;
                let z = z0 + zi as i64;
                let ground = ground_y[zi * w + xi];
                let height = self.tree_height(x, z);
                trees.push(Tree {
                    x,
                    z,
                    ground,
                    height,
                });
            }
        }
        trees
    }

    /// 位置 (x, z) 是否 roll 中树.
    fn roll(&self, x: i64, z: i64) -> bool {
        hash01(x, 0, z, self.seed ^ 0x77) < self.density
    }

    /// 树干高度: 4 或 5, 由位置 hash 决定.
    fn tree_height(&self, x: i64, z: i64) -> i64 {
        if hash01(x, 1, z, self.seed ^ 0x4B) < 0.5 {
            4
        } else {
            5
        }
    }
}

/// 高度是否对应草地表面 (可种树).
fn is_grass_surface(h: f64) -> bool {
    h > SEA_LEVEL + 2.0 && h <= SEA_LEVEL + 10.0
}

/// 复用 column.rs 同款 hash → [0, 1).
fn hash01(x: i64, y: i64, z: i64, seed: u64) -> f64 {
    let mut h = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    h = splitmix64(h.wrapping_add(x as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    h = splitmix64(h.wrapping_add(y as u64).wrapping_mul(0x94d0_49bb_1331_11eb));
    h = splitmix64(h.wrapping_add(z as u64).wrapping_mul(0xda94_9d13_b7dd_3787));
    ((h >> 11) as f64) / ((1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::continent::WorldHeightmap;

    #[test]
    fn tree_blocks_shape() {
        let t = Tree {
            x: 0,
            z: 0,
            ground: 64,
            height: 4,
        };
        let blocks = t.blocks();

        // 树干 4 格.
        let trunks: Vec<_> = blocks
            .iter()
            .filter(|(_, _, _, b)| *b == Block::OakLog)
            .collect();
        assert_eq!(trunks.len(), 4);
        for &(_, y, _, _) in &trunks {
            assert!((65i64..=68).contains(y), "trunk y={y}");
        }

        // 有叶子.
        let leaves: Vec<_> = blocks
            .iter()
            .filter(|(_, _, _, b)| *b == Block::OakLeaves)
            .collect();
        assert!(!leaves.is_empty());

        // 叶子不替换树干.
        for &(x, y, z, b) in &blocks {
            if b == Block::OakLog {
                assert!(
                    !leaves
                        .iter()
                        .any(|(lx, ly, lz, _)| *lx == x && *ly == y && *lz == z)
                );
            }
        }

        // 叶子距树干水平距离 <= 2, 且不切四角.
        for &(x, y, z, b) in &blocks {
            if b == Block::OakLeaves {
                let dx = x.abs();
                let dz = z.abs();
                assert!(!(dx == 2 && dz == 2), "corner leaf at ({x},{z})");
                assert!(dx <= 2 && dz <= 2, "leaf at ({x},{y},{z}) too far");
            }
        }
    }

    #[test]
    fn deterministic() {
        let world = WorldHeightmap::new(42);
        let d = TreeDecorator::new(42);
        let a = d.generate(&world, 0, 300, 0, 300);
        let b = d.generate(&world, 0, 300, 0, 300);
        assert_eq!(a.len(), b.len());
        for (ta, tb) in a.iter().zip(b.iter()) {
            assert_eq!(ta.x, tb.x);
            assert_eq!(ta.z, tb.z);
            assert_eq!(ta.height, tb.height);
        }
    }

    #[test]
    fn min_spacing() {
        // 用一个已知有陆地的 seed 和范围.
        let world = WorldHeightmap::new(2026_0904);
        let d = TreeDecorator::with_density(2026_0904, 0.15);
        let trees = d.generate(&world, 0, 2000, 0, 2000);
        assert!(!trees.is_empty(), "should have some trees at high density");
        for (i, a) in trees.iter().enumerate() {
            for (j, b) in trees.iter().enumerate() {
                if i == j {
                    continue;
                }
                let dx = (a.x - b.x) as f64;
                let dz = (a.z - b.z) as f64;
                let dist = (dx * dx + dz * dz).sqrt();
                assert!(
                    dist >= 3.0,
                    "trees ({},{}) and ({},{}) too close: {dist}",
                    a.x,
                    a.z,
                    b.x,
                    b.z
                );
            }
        }
    }

    #[test]
    fn trees_only_on_grass() {
        // 树只出现在 SEA_LEVEL+2 < h <= SEA_LEVEL+10 的区域.
        assert!(is_grass_surface(64.0));
        assert!(is_grass_surface(68.0));
        assert!(!is_grass_surface(60.0)); // 水
        assert!(!is_grass_surface(61.0)); // 沙滩
        assert!(!is_grass_surface(75.0)); // 高山
    }

    #[test]
    fn reasonable_tree_count() {
        let world = WorldHeightmap::new(2026_0904);
        let d = TreeDecorator::new(2026_0904);
        let trees = d.generate(&world, 0, 2000, 0, 2000);
        // 2000x2000 格, 4% 密度, 间距过滤后数量应合理.
        assert!(
            trees.len() > 50 && trees.len() < 20000,
            "tree count: {}",
            trees.len()
        );
    }
}
