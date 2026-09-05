//! 地形柱填充: 高度 `h` (格, f64) → `Y_MIN..=max(top, SEA_Y)` 的方块列.
//!
//! 纯函数, 与高度场解耦: 调用方传 `h = world.height(x, z)` 进来即可.
//! 分层 (y 从下往上):
//!
//! ```text
//! y=0            基岩
//! y=1..=top-4    石头 (按深度掺煤矿/铁/金/钻石, 见 ORE 表)
//! y=top-3..=top-1 垫层: 陆地=泥土, 沙滩/浅海=沙子, 深海=沙砾
//! y=top          表面: 陆地=草方块, 沙滩/浅海=沙子, 深海=沙砾 (小概率黏土透镜)
//! y=top+1..=60   水 (仅 h <= SEA_LEVEL 时)
//! ```
//!
//! - `SEA_Y = 60` = `floor(SEA_LEVEL=60.5)`. 水面占到 y=60.
//! - 沙滩: `SEA < h <= SEA+2` (top 为 61/62). 浅海: `SEA-4 <= h <= SEA`.
//! - 沙滩/浅海垫层下另有 2 格砂岩 (`top-5..=top-4` 为沙子区时).
//! - 矿石只替换石头区, 按一次 hash 互斥 roll (钻石>金>铁>煤).

use super::block::Block;
use super::continent::SEA_LEVEL;
use super::plains::splitmix64;

/// 最低/最高 y. `Y_MAX` 留出树/建筑空间, 柱只存到 `max(top, SEA_Y)`.
pub const Y_MIN: i64 = 0;
pub const Y_MAX: i64 = 127;
/// 水面 y (`floor(60.5)`).
pub const SEA_Y: i64 = 60;

pub struct ColumnGen {
    pub seed: u64,
}

#[derive(Clone)]
pub struct Column {
    pub x: i64,
    pub z: i64,
    /// `blocks[y - Y_MIN]`, 长度 `= max(top, SEA_Y) + 1`.
    pub blocks: Vec<Block>,
}

impl ColumnGen {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// 纯函数: 同一 `(x, z, h)` 必得同一柱.
    pub fn generate(&self, x: i64, z: i64, h: f64) -> Column {
        let top = (h.floor() as i64).clamp(Y_MIN, Y_MAX);
        let stored_top = top.max(SEA_Y);
        let mut blocks = vec![Block::Air; (stored_top - Y_MIN + 1) as usize];
        let set = |blocks: &mut Vec<Block>, y: i64, b: Block| {
            blocks[(y - Y_MIN) as usize] = b;
        };

        let is_land = h > SEA_LEVEL;
        let is_beach = is_land && h <= SEA_LEVEL + 2.0;
        let is_shallow = !is_land && h >= SEA_LEVEL - 4.0;

        // 表面块.
        let surface = if is_land && !is_beach {
            Block::GrassBlock
        } else if is_beach || is_shallow {
            Block::Sand
        } else if hash01(x, 0, z, self.seed ^ 0xC1A9) < 0.08 {
            // 深海小概率黏土透镜.
            Block::Clay
        } else {
            Block::Gravel
        };

        for y in Y_MIN..=stored_top {
            let b = if y == Y_MIN {
                Block::Bedrock
            } else if y > top {
                Block::Water
            } else if y == top {
                surface
            } else if y >= top - 3 {
                match surface {
                    Block::GrassBlock => Block::Dirt,
                    Block::Sand => Block::Sand,
                    _ => Block::Gravel,
                }
            } else if y >= top - 5 && (surface == Block::Sand) {
                Block::Sandstone
            } else {
                self.stone_or_ore(x, y, z)
            };
            set(&mut blocks, y, b);
        }
        Column { x, z, blocks }
    }

    fn stone_or_ore(&self, x: i64, y: i64, z: i64) -> Block {
        let r = hash01(x, y, z, self.seed);
        if y <= 16 {
            if r < 0.008 {
                Block::DiamondOre
            } else if r < 0.013 {
                Block::GoldOre
            } else if r < 0.028 {
                Block::IronOre
            } else if r < 0.050 {
                Block::CoalOre
            } else {
                Block::Stone
            }
        } else if y <= 32 {
            if r < 0.006 {
                Block::GoldOre
            } else if r < 0.020 {
                Block::IronOre
            } else if r < 0.045 {
                Block::CoalOre
            } else {
                Block::Stone
            }
        } else if y <= 64 {
            if r < 0.012 {
                Block::IronOre
            } else if r < 0.035 {
                Block::CoalOre
            } else {
                Block::Stone
            }
        } else if r < 0.020 {
            Block::CoalOre
        } else {
            Block::Stone
        }
    }
}

impl Column {
    pub fn get(&self, y: i64) -> Block {
        if y < Y_MIN || y > Y_MIN + self.blocks.len() as i64 - 1 {
            return Block::Air;
        }
        self.blocks[(y - Y_MIN) as usize]
    }

    /// 最高固体 y，流体和空气都不算作地面。
    pub fn top_solid(&self) -> i64 {
        for y in (Y_MIN..Y_MIN + self.blocks.len() as i64).rev() {
            let b = self.get(y);
            if b.is_solid() {
                return y;
            }
        }
        Y_MIN
    }
}

/// 确定性 hash → [0, 1). 同一输入必同一输出, 与坐标无关的均匀性不作保证,
/// 只用于矿石/黏土这类"有即可"的装饰分布.
fn hash01(x: i64, y: i64, z: i64, seed: u64) -> f64 {
    let mut h = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    h = splitmix64(h.wrapping_add(x as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    h = splitmix64(h.wrapping_add(y as u64).wrapping_mul(0x94d0_49bb_1331_11eb));
    h = splitmix64(h.wrapping_add(z as u64).wrapping_mul(0xda94_9d13_b7dd_3787));
    // 取高 53 位 → [0,1).
    ((h >> 11) as f64) / ((1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::world::continent::WorldHeightmap;

    #[test]
    fn deterministic() {
        let g = ColumnGen::new(99);
        let a = g.generate(12, -34, 64.2);
        let b = g.generate(12, -34, 64.2);
        assert_eq!(a.blocks, b.blocks);
    }

    #[test]
    fn layers_land() {
        let g = ColumnGen::new(7);
        let c = g.generate(0, 0, 70.0);
        assert_eq!(c.get(0), Block::Bedrock);
        assert_eq!(c.get(70), Block::GrassBlock);
        assert_eq!(c.get(69), Block::Dirt);
        assert_eq!(c.get(68), Block::Dirt);
        assert_eq!(c.get(67), Block::Dirt);
        // 垫层下是石头/矿石/砂岩之外的固体, 且绝不出现水.
        assert_eq!(c.get(71), Block::Air);
        assert_eq!(c.top_solid(), 70);
    }

    #[test]
    fn beach_and_sea() {
        let g = ColumnGen::new(7);
        // 沙滩 top=61.
        let b = g.generate(5, 5, 61.3);
        assert_eq!(b.get(61), Block::Sand);
        assert_eq!(b.get(60), Block::Sand);
        // 海: h=55 → top=55 为浅海沙, 56..=60 全水.
        let s = g.generate(5, 6, 58.0);
        assert_eq!(s.get(58), Block::Sand);
        for y in 59..=SEA_Y {
            assert_eq!(s.get(y), Block::Water, "y={y} 应为水");
        }
        assert_eq!(s.top_solid(), 58);
    }

    #[test]
    fn solid_contiguous_no_air_pockets() {
        // 0..=top 必须全固体 (基岩/石头/垫层/表面), 不许有空气/水夹心.
        let g = ColumnGen::new(1234);
        for h in [44.0, 55.0, 60.4, 60.6, 62.0, 70.0, 100.0] {
            let c = g.generate(11, -7, h);
            let top = h.floor() as i64;
            for y in Y_MIN..=top {
                let b = c.get(y);
                assert!(
                    b != Block::Air && b != Block::Water,
                    "h={h} y={y} 出现空洞: {b:?}"
                );
            }
        }
    }

    #[test]
    fn ores_only_in_stone_zone() {
        let g = ColumnGen::new(42);
        let c = g.generate(3, 9, 90.0);
        for y in 1..=90 {
            let b = c.get(y);
            let is_ore = matches!(
                b,
                Block::CoalOre | Block::IronOre | Block::GoldOre | Block::DiamondOre
            );
            if is_ore {
                assert!(y <= 90 - 6, "矿石只能在石头区: y={y}");
            }
            // 垫层/表面绝不出现矿石.
            if y >= 90 - 3 {
                assert!(!is_ore, "垫层/表面出现矿石: y={y}");
            }
        }
    }

    #[test]
    fn world_integration_counts() {
        // 真实高度场上跑 256 列: 陆地/海洋必须都有, 水只在海平面以下.
        let w = WorldHeightmap::new(2026_0904);
        let g = ColumnGen::new(2026_0904);
        let mut land = 0;
        let mut sea = 0;
        let mut counts: HashMap<Block, u32> = HashMap::new();
        for i in 0..16 {
            for j in 0..16 {
                let (x, z) = (i * 997 - 8000, j * 761 + 3000);
                let h = w.height(x, z);
                let c = g.generate(x, z, h);
                *counts.entry(c.get(c.top_solid())).or_insert(0) += 1;
                if h > SEA_LEVEL {
                    land += 1;
                    // 陆地柱无水, 顶部之上是空气.
                    let top = h.floor() as i64;
                    assert_eq!(c.get(top + 1), Block::Air);
                } else {
                    sea += 1;
                    // 真海 (top < SEA_Y) 水面封顶; 潮间带 (top == SEA_Y) 为出露海床.
                    let top = h.floor() as i64;
                    if top < SEA_Y {
                        assert_eq!(c.get(SEA_Y), Block::Water);
                    }
                }
            }
        }
        assert!(land > 0 && sea > 0, "陆地/海洋必须同时存在");
        assert!(counts.contains_key(&Block::GrassBlock));
        assert!(counts.contains_key(&Block::Sand) || counts.contains_key(&Block::Gravel));
    }
}
