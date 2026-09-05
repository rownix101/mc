//! 体素世界查询适配器。
//!
//! 物理层只依赖 `VoxelWorld::block_at`，因此将来替换成 chunk streaming
//! 不需要改动碰撞算法。当前实现按需从确定性高度图生成并缓存柱。

use std::collections::HashMap;

use rayon::prelude::*;

use super::block::Block;
use super::column::{Column, ColumnGen, SEA_Y, Y_MAX, Y_MIN};
use super::continent::WorldHeightmap;

/// 物理引擎需要的最小世界接口。
pub trait VoxelWorld {
    fn block_at(&mut self, x: i64, y: i64, z: i64) -> Block;

    fn set_block(&mut self, _x: i64, _y: i64, _z: i64, _block: Block) {}
}

/// 基于当前世界生成器的按需体素查询。
pub struct GeneratedVoxelWorld {
    heightmap: WorldHeightmap,
    column_gen: ColumnGen,
    columns: HashMap<(i64, i64), Column>,
    /// Player edits override the deterministic generator, including edits to air.
    edits: HashMap<(i64, i64, i64), Block>,
    /// Highest edited y per column. This keeps mesh bounds queries O(1) instead
    /// of scanning every edit for every column in the visible region.
    edit_column_tops: HashMap<(i64, i64), i64>,
}

impl GeneratedVoxelWorld {
    pub fn new(seed: u64) -> Self {
        Self::with_columns(seed, std::iter::empty())
    }

    /// Create a world with deterministic columns that were already generated
    /// by the world-creation step. This avoids regenerating the spawn area
    /// when the player first enters the world.
    pub fn with_columns(seed: u64, columns: impl IntoIterator<Item = Column>) -> Self {
        Self {
            heightmap: WorldHeightmap::new(seed),
            column_gen: ColumnGen::new(seed),
            columns: columns
                .into_iter()
                .map(|column| ((column.x, column.z), column))
                .collect(),
            edits: HashMap::new(),
            edit_column_tops: HashMap::new(),
        }
    }

    pub fn cached_column_count(&self) -> usize {
        self.columns.len()
    }

    pub fn cached_columns(&self) -> impl Iterator<Item = &Column> {
        self.columns.values()
    }

    /// Generate all columns in a rectangular x/z range that are not cached.
    /// Generation is independent per column; only the final HashMap insertion
    /// remains serial, keeping the persistent cache usable between mesh jobs.
    pub fn preload_columns_parallel(&mut self, min_x: i64, max_x: i64, min_z: i64, max_z: i64) {
        let missing: Vec<_> = (min_z..=max_z)
            .flat_map(|z| (min_x..=max_x).map(move |x| (x, z)))
            .filter(|coordinate| !self.columns.contains_key(coordinate))
            .collect();
        let generated: Vec<_> = missing
            .into_par_iter()
            .map(|(x, z)| {
                let height = self.heightmap.height(x, z);
                self.column_gen.generate(x, z, height)
            })
            .collect();
        self.columns.extend(
            generated
                .into_iter()
                .map(|column| ((column.x, column.z), column)),
        );
    }

    pub fn edits(&self) -> impl Iterator<Item = (&(i64, i64, i64), &Block)> {
        self.edits.iter()
    }

    /// 返回该列可能包含方块的最高 y。
    ///
    /// 陆地需要扫描到地表，海洋还要扫描到水面；更高的位置一定是空气，
    /// 网格生成器可以据此跳过大量无意义的查询。
    pub fn surface_top(&self, x: i64, z: i64) -> i64 {
        let generated_top = self
            .heightmap
            .height(x, z)
            .floor()
            .clamp(Y_MIN as f64, Y_MAX as f64)
            .max(SEA_Y as f64) as i64;
        let edited_top = self.edit_column_tops.get(&(x, z)).copied().unwrap_or(Y_MIN);
        generated_top.max(edited_top)
    }
}

impl VoxelWorld for GeneratedVoxelWorld {
    fn block_at(&mut self, x: i64, y: i64, z: i64) -> Block {
        if !(Y_MIN..=Y_MAX).contains(&y) {
            return Block::Air;
        }

        if let Some(&block) = self.edits.get(&(x, y, z)) {
            return block;
        }

        if let Some(column) = self.columns.get(&(x, z)) {
            return column.get(y);
        }

        let height = self.heightmap.height(x, z);
        let column = self.column_gen.generate(x, z, height);
        self.columns.insert((x, z), column);
        self.columns
            .get(&(x, z))
            .expect("刚插入的体素柱必须存在")
            .get(y)
    }

    fn set_block(&mut self, x: i64, y: i64, z: i64, block: Block) {
        if (Y_MIN..=Y_MAX).contains(&y) {
            self.edits.insert((x, y, z), block);
            self.edit_column_tops
                .entry((x, z))
                .and_modify(|top| *top = (*top).max(y))
                .or_insert(y);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_world_is_deterministic_and_caches_columns() {
        let mut a = GeneratedVoxelWorld::new(1234);
        let mut b = GeneratedVoxelWorld::new(1234);

        assert_eq!(a.block_at(4, 70, -3), b.block_at(4, 70, -3));
        assert_eq!(a.cached_column_count(), 1);
        assert_eq!(a.block_at(4, 71, -3), b.block_at(4, 71, -3));
        assert_eq!(a.cached_column_count(), 1);
        assert_eq!(a.block_at(5, 70, -3), b.block_at(5, 70, -3));
        assert_eq!(a.cached_column_count(), 2);
    }

    #[test]
    fn outside_vertical_range_is_air() {
        let mut world = GeneratedVoxelWorld::new(7);
        assert_eq!(world.block_at(0, Y_MIN - 1, 0), Block::Air);
        assert_eq!(world.block_at(0, Y_MAX + 1, 0), Block::Air);
        assert_eq!(world.cached_column_count(), 0);
    }

    #[test]
    fn edits_override_generated_blocks_and_persist_as_air() {
        let mut world = GeneratedVoxelWorld::new(7);
        let original = world.block_at(0, 60, 0);
        world.set_block(0, 60, 0, Block::OakPlanks);
        assert_eq!(world.block_at(0, 60, 0), Block::OakPlanks);
        world.set_block(0, 60, 0, Block::Air);
        assert_eq!(world.block_at(0, 60, 0), Block::Air);
        assert_ne!(original, Block::OakPlanks);
    }
}
