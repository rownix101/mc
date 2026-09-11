//! 树木装饰器: 在草地表面概率性放置橡树.
//!
//! 与 `ColumnGen` 正交: `ColumnGen` 产出纯地形柱, `TreeDecorator` 扫描一个区域、
//! 挑候选位置、输出 `Vec<Tree>`. 调用方把 `Tree::blocks()` 写入已有柱即可.
//! 不引入新的 Chunk 结构.
//!
//! 橡树使用确定性的位置 hash 变化: 树干高 4--7 格, 树冠 3--4 层,
//! 冠幅沿两个轴独立选择半径 1 或 2, 外沿有局部缺口; 高树有短的根部加粗.
//! 所有方块保持在主干水平半径 2 格内, 与逐列装饰和 LOD 查询边界一致.
//! 放置条件: 草地表面 (SEA_LEVEL+2 < h <= SEA_LEVEL+10).
//! 最小主干间距 3 格 (树冠仍可重叠), 默认候选概率 0.04.

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

        let shape = hash64(self.x, self.height, self.z, 0x715e_ed0a);
        let radius_x = if shape & 3 == 0 { 1_i64 } else { 2 };
        let radius_z = if (shape >> 2) & 3 == 0 { 1_i64 } else { 2 };
        let bottom = if self.height >= 6 { -2_i64 } else { -1 };
        let gaps = 0.10 + ((shape >> 4) & 3) as f64 * 0.05;
        // Tall trees get a short buttress, still within the existing query halo.
        if self.height >= 6 {
            let (dx, dz) = match (shape >> 6) & 3 {
                0 => (1, 0),
                1 => (-1, 0),
                2 => (0, 1),
                _ => (0, -1),
            };
            out.push((self.x + dx, top, self.z + dz, Block::OakLog));
            if self.height == 7 {
                out.push((self.x + dx, top + 1, self.z + dz, Block::OakLog));
            }
        }
        let canopy_center = top + self.height - 1;
        for dy in bottom..=1_i64 {
            let y = canopy_center + dy;
            let rx = if dy == 1 { 1 } else { radius_x };
            let rz = if dy == 1 { 1 } else { radius_z };
            for dx in -rx..=rx {
                for dz in -rz..=rz {
                    if dx == 0 && dz == 0 && dy <= 0 {
                        continue;
                    }
                    if dx.abs() == rx && dz.abs() == rz {
                        continue;
                    }
                    let edge = dx.abs() == rx || dz.abs() == rz;
                    if edge && hash01(self.x + dx, dy + 16, self.z + dz, shape) < gaps {
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
        debug_assert!(
            x0 <= x1 && z0 <= z1,
            "generate 的区域必须是半开区间 x0<=x1, z0<=z1"
        );
        let w = (x1 - x0).max(0) as usize;
        let h = (z1 - z0).max(0) as usize;
        if w == 0 || h == 0 {
            return Vec::new();
        }

        // 保留 2 格 padding，这样每个候选的 spacing 胜负只依赖局部
        // ±2 邻域，不依赖调用方一次查询了多大区域、也不依赖扫描顺序。
        const PAD: i64 = 2;
        let padded_x0 = x0 - PAD;
        let padded_z0 = z0 - PAD;
        let padded_w = w + (PAD * 2) as usize;
        let padded_h = h + (PAD * 2) as usize;

        let mut candidates = vec![false; padded_w * padded_h];
        let mut priorities = vec![0u64; padded_w * padded_h];
        let mut ground_y = vec![0i64; padded_w * padded_h];

        for pzi in 0..padded_h {
            let z = padded_z0 + pzi as i64;
            for pxi in 0..padded_w {
                let x = padded_x0 + pxi as i64;
                let hi = world.height(x, z);
                let index = pzi * padded_w + pxi;
                ground_y[index] = hi.floor() as i64;
                if is_grass_surface(hi) && self.roll(x, z) {
                    candidates[index] = true;
                    priorities[index] = self.priority(x, z);
                }
            }
        }

        let mut trees = Vec::new();
        for zi in 0..h {
            let z = z0 + zi as i64;
            for xi in 0..w {
                let x = x0 + xi as i64;
                let pxi = xi + PAD as usize;
                let pzi = zi + PAD as usize;
                let index = pzi * padded_w + pxi;
                if !candidates[index] {
                    continue;
                }

                let priority = priorities[index];
                let mut winner = true;
                for dz in -PAD..=PAD {
                    for dx in -PAD..=PAD {
                        if dx == 0 && dz == 0 {
                            continue;
                        }
                        if dx * dx + dz * dz >= 9 {
                            continue;
                        }
                        let nxi = pxi as i64 + dx;
                        let nzi = pzi as i64 + dz;
                        let nindex = nzi as usize * padded_w + nxi as usize;
                        if !candidates[nindex] {
                            continue;
                        }
                        let neighbour_priority = priorities[nindex];
                        if neighbour_priority > priority
                            || (neighbour_priority == priority && (x + dx, z + dz) < (x, z))
                        {
                            winner = false;
                            break;
                        }
                    }
                    if !winner {
                        break;
                    }
                }

                if winner {
                    trees.push(Tree {
                        x,
                        z,
                        ground: ground_y[index],
                        height: self.tree_height(x, z),
                    });
                }
            }
        }

        trees
    }

    /// 位置 (x, z) 是否 roll 中树.
    fn roll(&self, x: i64, z: i64) -> bool {
        hash01(x, 0, z, self.seed ^ 0x77) < self.density
    }

    /// 候选树的确定性优先级。spacing 过滤只保留局部邻域内优先级最高的候选，
    /// 因此结果不依赖扫描顺序或查询窗口大小。
    fn priority(&self, x: i64, z: i64) -> u64 {
        hash64(x, 3, z, self.seed ^ 0x5A17_5A17_5A17_5A17)
    }

    /// 树干高度: 4--7 格, 高树较少, 由 seed 和位置 hash 决定.
    fn tree_height(&self, x: i64, z: i64) -> i64 {
        let roll = hash01(x, 2, z, self.seed ^ 0x4B);
        if roll < 0.25 {
            4
        } else if roll < 0.62 {
            5
        } else if roll < 0.90 {
            6
        } else {
            7
        }
    }
}

/// 高度是否对应草地表面 (可种树).
fn is_grass_surface(h: f64) -> bool {
    h > SEA_LEVEL + 2.0 && h <= SEA_LEVEL + 10.0
}

/// 复用 column.rs 同款 hash → [0, 1).
fn hash64(x: i64, y: i64, z: i64, seed: u64) -> u64 {
    let mut h = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    h = splitmix64(h.wrapping_add(x as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    h = splitmix64(h.wrapping_add(y as u64).wrapping_mul(0x94d0_49bb_1331_11eb));
    h = splitmix64(h.wrapping_add(z as u64).wrapping_mul(0xda94_9d13_b7dd_3787));
    h
}

fn hash01(x: i64, y: i64, z: i64, seed: u64) -> f64 {
    ((hash64(x, y, z, seed) >> 11) as f64) / ((1u64 << 53) as f64)
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
    fn varied_shapes_stay_inside_column_query_halo() {
        use std::collections::HashSet;
        let mut silhouettes = HashSet::new();
        for x in -32..32 {
            for height in 4..=7 {
                let tree = Tree {
                    x,
                    z: -17,
                    ground: 64,
                    height,
                };
                let blocks = tree.blocks();
                assert_eq!(blocks, tree.blocks());
                let mut occupied = HashSet::new();
                for &(bx, by, bz, _) in &blocks {
                    assert!((bx - x).abs() <= 2 && (bz + 17).abs() <= 2);
                    assert!((65..=65 + height).contains(&by));
                    assert!(occupied.insert((bx, by, bz)), "overlapping wood/leaves");
                }
                silhouettes.insert(
                    blocks
                        .iter()
                        .filter_map(|&(bx, by, bz, block)| {
                            (block == Block::OakLeaves).then_some((bx - x, by - height, bz + 17))
                        })
                        .collect::<Vec<_>>(),
                );
            }
        }
        assert!(silhouettes.len() > 32, "canopies should visibly vary");
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
    fn local_queries_agree_with_global_generation() {
        let seed = 2026_0904;
        let world = WorldHeightmap::new(seed);
        let d = TreeDecorator::new(seed);
        let global = d.generate(&world, 400, 500, 900, 1000);
        assert!(!global.is_empty(), "测试区域应至少有一棵树");

        for tree in global {
            // decorate_column 每列只查周围 2 格，因此用同样的局部窗口
            // 重新生成时，全局被保留的树也必须再次出现。
            let local = d.generate(&world, tree.x - 2, tree.x + 3, tree.z - 2, tree.z + 3);
            assert!(
                local.iter().any(|candidate| candidate.x == tree.x
                    && candidate.z == tree.z
                    && candidate.height == tree.height),
                "树 ({}, {}) 在大范围生成中存在，但在局部 5x5 查询中丢失",
                tree.x,
                tree.z
            );
        }
    }

    #[test]
    fn tree_heights_cover_varied_range() {
        let world = WorldHeightmap::new(2026_0904);
        let d = TreeDecorator::with_density(2026_0904, 0.15);
        let trees = d.generate(&world, 0, 1200, 0, 1200);
        assert!(!trees.is_empty());

        for expected in [4, 5, 6, 7] {
            assert!(
                trees.iter().any(|tree| tree.height == expected),
                "没有生成高度为 {expected} 的树"
            );
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
