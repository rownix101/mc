//! 体素光照: 在稠密方块快照上做洪泛 (flood-fill), 作为 GI 的低频间接光基础.
//!
//! 这一层只传播"直接可见的天空光 / 发光方块", 不含多次反弹; 结果按顶点
//! 平滑后作为天空可见度送进前向着色器. 后续世界空间辐照探针会复用同一套
//! 可见性查询.
//!
//! - 天空光: `y > 该列最高非空气方块` 的格子初始为 [`LIGHT_MAX`] (完全见天),
//!   再从这些源向被遮挡区域做多源 BFS; 每进入一格按
//!   [`Block::light_attenuation`] 衰减.
//! - 方块光: 从 [`Block::light_emission`] 非零的格子出发, 规则相同.
//!   当前注册表还没有发光方块, 接入火把 / 萤石后无需改这里.
//!
//! 数组布局与网格器的稠密快照一致: `(z * width + x) * height + y`.

use std::collections::VecDeque;

use super::block::{Block, LIGHT_MAX};

/// 在稠密方块快照上计算天空光.
///
/// `max_top` 是区域内最高的列顶 (数组下标). `max_top + 1` 以上没有几何,
/// 调用方应把那里的查询直接当成完全见天, 因此 BFS 只需覆盖到该高度, 不必
/// 把整根开放空气柱都塞进队列.
pub fn flood_fill_sky(
    blocks: &[Block],
    width: usize,
    depth: usize,
    height: usize,
    max_top: usize,
) -> Vec<u8> {
    let mut light = vec![0u8; width * depth * height];
    let mut queue = VecDeque::new();
    let seed_top = (max_top + 1).min(height - 1);

    for z in 0..depth {
        for x in 0..width {
            let base = (z * width + x) * height;
            let mut top = None;
            for y in (0..=seed_top).rev() {
                if blocks[base + y] != Block::Air {
                    top = Some(y);
                    break;
                }
            }
            for y in top.map_or(0, |top| top + 1)..=seed_top {
                light[base + y] = LIGHT_MAX;
                queue.push_back(base + y);
            }
        }
    }

    propagate(&mut light, blocks, width, depth, height, &mut queue);
    light
}

/// 在稠密方块快照上计算方块光 (自发光方块的多源洪泛).
pub fn flood_fill_block(blocks: &[Block], width: usize, depth: usize, height: usize) -> Vec<u8> {
    let mut light = vec![0u8; width * depth * height];
    let mut queue = VecDeque::new();
    for (index, &block) in blocks.iter().enumerate() {
        let emission = block.light_emission();
        if emission > 0 {
            light[index] = emission;
            queue.push_back(index);
        }
    }
    propagate(&mut light, blocks, width, depth, height, &mut queue);
    light
}

/// 六向多源 BFS. 队首的格点只会把光传给更暗的邻居, 因此每个格点最多按
/// 自身亮度被松弛一次; 不透明方块直接跳过.
fn propagate(
    light: &mut [u8],
    blocks: &[Block],
    width: usize,
    depth: usize,
    height: usize,
    queue: &mut VecDeque<usize>,
) {
    while let Some(index) = queue.pop_front() {
        let level = light[index];
        if level <= 1 {
            continue;
        }
        let y = index % height;
        let column = index / height;
        let x = column % width;
        let z = column / width;
        if y > 0 {
            relax(light, blocks, queue, index - 1, level);
        }
        if y + 1 < height {
            relax(light, blocks, queue, index + 1, level);
        }
        if x > 0 {
            relax(light, blocks, queue, index - height, level);
        }
        if x + 1 < width {
            relax(light, blocks, queue, index + height, level);
        }
        if z > 0 {
            relax(light, blocks, queue, index - width * height, level);
        }
        if z + 1 < depth {
            relax(light, blocks, queue, index + width * height, level);
        }
    }
}

fn relax(
    light: &mut [u8],
    blocks: &[Block],
    queue: &mut VecDeque<usize>,
    target: usize,
    level: u8,
) {
    let attenuation = blocks[target].light_attenuation();
    if attenuation > LIGHT_MAX {
        return;
    }
    let candidate = level.saturating_sub(attenuation);
    if candidate > light[target] {
        light[target] = candidate;
        queue.push_back(target);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(light: &[u8], width: usize, height: usize, x: usize, y: usize, z: usize) -> u8 {
        light[(z * width + x) * height + y]
    }

    #[test]
    fn open_column_is_fully_lit() {
        let (width, depth, height) = (1, 1, 4);
        let blocks = vec![Block::Air; width * depth * height];
        let light = flood_fill_sky(&blocks, width, depth, height, height - 1);
        for y in 0..height {
            assert_eq!(at(&light, width, height, 0, y, 0), LIGHT_MAX);
        }
    }

    #[test]
    fn solid_column_is_dark_below_the_surface() {
        let (width, depth, height) = (1, 1, 4);
        // y=0/1 stone, y=2/3 open sky.
        let blocks = vec![Block::Stone, Block::Stone, Block::Air, Block::Air];
        let light = flood_fill_sky(&blocks, width, depth, height, height - 1);
        assert_eq!(at(&light, width, height, 0, 0, 0), 0);
        assert_eq!(at(&light, width, height, 0, 1, 0), 0);
        assert_eq!(at(&light, width, height, 0, 2, 0), LIGHT_MAX);
        assert_eq!(at(&light, width, height, 0, 3, 0), LIGHT_MAX);
    }

    #[test]
    fn overhang_shadow_falls_off_laterally() {
        // x=0 is an open air column; x=1/2 have a two-block stone roof on top.
        let (width, depth, height) = (3, 1, 6);
        let mut blocks = vec![Block::Air; width * depth * height];
        for x in 1..3 {
            for y in 4..6 {
                blocks[x * height + y] = Block::Stone;
            }
        }
        let light = flood_fill_sky(&blocks, width, depth, height, height - 1);
        // The open column is the source.
        assert_eq!(at(&light, width, height, 0, 0, 0), LIGHT_MAX);
        // Light reaches the shaded cells sideways and fades with distance.
        let near = at(&light, width, height, 1, 0, 0);
        let far = at(&light, width, height, 2, 0, 0);
        assert!(
            near < LIGHT_MAX,
            "roofed cell must not be fully lit: {near}"
        );
        assert!(
            far < near,
            "light must fall off with distance: {far} < {near}"
        );
    }

    #[test]
    fn propagation_attenuates_by_one_per_air_cell() {
        let (width, depth, height) = (5, 1, 1);
        let blocks = vec![Block::Air; width * depth * height];
        let mut light = vec![0u8; width * depth * height];
        light[0] = LIGHT_MAX;
        let mut queue = VecDeque::from([0]);
        propagate(&mut light, &blocks, width, depth, height, &mut queue);
        assert_eq!(light, vec![15, 14, 13, 12, 11]);
    }

    #[test]
    fn water_and_leaves_attenuate_faster_than_air() {
        let (width, depth, height) = (3, 1, 1);
        let blocks = vec![Block::Air, Block::Water, Block::OakLeaves];
        let mut light = vec![0u8; width];
        light[0] = LIGHT_MAX;
        let mut queue = VecDeque::from([0]);
        propagate(&mut light, &blocks, width, depth, height, &mut queue);
        // Air costs 1, water costs 2, leaves cost 2.
        assert_eq!(light, vec![15, 13, 11]);
    }

    #[test]
    fn opaque_blocks_stop_propagation() {
        let (width, depth, height) = (3, 1, 1);
        let blocks = vec![Block::Air, Block::Stone, Block::Air];
        let mut light = vec![0u8; width];
        light[0] = LIGHT_MAX;
        let mut queue = VecDeque::from([0]);
        propagate(&mut light, &blocks, width, depth, height, &mut queue);
        assert_eq!(light, vec![15, 0, 0]);
    }

    #[test]
    fn block_light_is_dark_until_emissive_blocks_exist() {
        let (width, depth, height) = (2, 1, 2);
        let blocks = vec![Block::Stone; width * depth * height];
        let light = flood_fill_block(&blocks, width, depth, height);
        assert!(light.iter().all(|&value| value == 0));
    }
}
