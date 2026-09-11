//! 体素世界查询适配器。
//!
//! 物理层只依赖 `VoxelWorld::block_at`，因此将来替换成 chunk streaming
//! 不需要改动碰撞算法。当前实现按需从确定性高度图生成并缓存柱。

use std::collections::HashMap;

use rayon::prelude::*;

use super::block::{Block, LIGHT_MAX};
use super::column::{Column, ColumnGen, SEA_Y, Y_MAX, Y_MIN};
use super::continent::WorldHeightmap;
use super::tree::{Tree, TreeDecorator};

/// 物理引擎需要的最小世界接口。
pub trait VoxelWorld {
    fn block_at(&mut self, x: i64, y: i64, z: i64) -> Block;

    fn set_block(&mut self, _x: i64, _y: i64, _z: i64, _block: Block) {}

    /// 平滑光照用的天空可见度 `0..=LIGHT_MAX`. 默认视为完全见天, 让不计算
    /// 光照的调用方 (物理世界 / 粗 LOD 路径) 保持原来的亮度.
    fn sky_light_at(&mut self, _x: i64, _y: i64, _z: i64) -> u8 {
        LIGHT_MAX
    }

    /// 平滑光照用的方块光 `0..=LIGHT_MAX`. 默认无发光方块.
    fn block_light_at(&mut self, _x: i64, _y: i64, _z: i64) -> u8 {
        0
    }
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
        let heightmap = WorldHeightmap::new(seed);
        let column_gen = ColumnGen::new(seed);
        let treegen = TreeDecorator::new(seed);
        let columns: Vec<Column> = columns
            .into_iter()
            .map(|column| decorate_column(column, &heightmap, &treegen))
            .collect();
        Self {
            heightmap,
            column_gen,
            columns: columns
                .into_iter()
                .map(|column| ((column.x, column.z), column))
                .collect(),
            edits: HashMap::new(),
            edit_column_tops: HashMap::new(),
        }
    }

    pub fn seed(&self) -> u64 {
        self.column_gen.seed
    }

    pub fn cached_column_count(&self) -> usize {
        self.columns.len()
    }

    pub fn cached_columns(&self) -> impl Iterator<Item = &Column> {
        self.columns.values()
    }

    /// Drop cached columns outside a square around `(center_x, center_z)`.
    ///
    /// The mesh worker uses this to keep its persistent column cache bounded
    /// while the player walks; generated columns are deterministic and cheap
    /// to rebuild if the player returns. The physics world is a separate
    /// instance and keeps its own cache.
    pub fn retain_columns_within(&mut self, center_x: i64, center_z: i64, radius: i64) {
        if radius < 0 {
            self.columns.clear();
            return;
        }
        self.columns
            .retain(|(x, z), _| (x - center_x).abs() <= radius && (z - center_z).abs() <= radius);
    }

    /// Generate all columns in a rectangular x/z range that are not cached.
    /// Generation is independent per column; only the final HashMap insertion
    /// remains serial, keeping the persistent cache usable between mesh jobs.
    pub fn preload_columns_parallel(&mut self, min_x: i64, max_x: i64, min_z: i64, max_z: i64) {
        let missing: Vec<_> = (min_z..=max_z)
            .flat_map(|z| (min_x..=max_x).map(move |x| (x, z)))
            .filter(|coordinate| !self.columns.contains_key(coordinate))
            .collect();
        let treegen = TreeDecorator::new(self.column_gen.seed);
        let generated: Vec<_> = missing
            .into_par_iter()
            .map(|(x, z)| {
                let terrain = self.heightmap.sample(x, z);
                let column = self.column_gen.generate_with_terrain(x, z, terrain);
                decorate_column(column, &self.heightmap, &treegen)
            })
            .collect();
        self.columns.extend(
            generated
                .into_iter()
                .map(|column| ((column.x, column.z), column)),
        );
    }

    /// Generate columns on a coarse grid anchored at `(center_x, center_z)`.
    ///
    /// Far LOD rings only need one representative column for every `step`
    /// blocks. Generating the full 1-block grid and throwing most of it away
    /// would defeat the point of the LOD, so this keeps the persistent cache
    /// proportional to the coarse ring instead.
    pub fn preload_columns_grid_parallel(
        &mut self,
        center_x: i64,
        center_z: i64,
        radius: i64,
        step: i64,
    ) {
        debug_assert!(step > 0, "LOD grid step must be positive");
        if radius < 0 {
            return;
        }
        let cells = radius / step;
        let missing: Vec<_> = (-cells..=cells)
            .flat_map(|kz| {
                (-cells..=cells).map(move |kx| (center_x + kx * step, center_z + kz * step))
            })
            .filter(|coordinate| !self.columns.contains_key(coordinate))
            .collect();
        let treegen = TreeDecorator::new(self.column_gen.seed);
        let generated: Vec<_> = missing
            .into_par_iter()
            .map(|(x, z)| {
                let terrain = self.heightmap.sample(x, z);
                let column = self.column_gen.generate_with_terrain(x, z, terrain);
                decorate_column(column, &self.heightmap, &treegen)
            })
            .collect();
        self.columns.extend(
            generated
                .into_iter()
                .map(|column| ((column.x, column.z), column)),
        );
    }

    /// Ensure every column in the rectangle exists and return owned clones.
    ///
    /// Near-chunk meshing copies these decorated columns into a dense snapshot
    /// and overlays edits there, preserving the persistent generator cache.
    pub fn clone_columns_region(
        &mut self,
        min_x: i64,
        max_x: i64,
        min_z: i64,
        max_z: i64,
    ) -> Vec<Column> {
        self.preload_columns_parallel(min_x, max_x, min_z, max_z);
        let mut out =
            Vec::with_capacity(((max_x - min_x + 1).max(0) * (max_z - min_z + 1).max(0)) as usize);
        for z in min_z..=max_z {
            for x in min_x..=max_x {
                if let Some(column) = self.columns.get(&(x, z)) {
                    out.push(column.clone());
                }
            }
        }
        out
    }

    /// Highest non-air block in a column, generating and caching it if needed.
    pub fn surface_block(&mut self, x: i64, z: i64) -> Option<(i64, Block)> {
        if !self.columns.contains_key(&(x, z)) {
            // Trigger the normal lazy generation path.
            let _ = self.block_at(x, Y_MIN, z);
        }
        let top = self.surface_top(x, z).clamp(Y_MIN, Y_MAX);
        for y in (Y_MIN..=top).rev() {
            let block = self.block_at(x, y, z);
            if block != Block::Air {
                return Some((y, block));
            }
        }
        None
    }

    /// Highest non-air terrain block in a column, ignoring generated trees.
    ///
    /// The coarse LOD mesher uses this so a tree canopy is never mistaken for
    /// the ground surface. Trees are submitted separately at full block
    /// resolution by [`Self::tree_blocks_in_region`].
    pub fn terrain_surface_block(&mut self, x: i64, z: i64) -> Option<(i64, Block)> {
        if !self.columns.contains_key(&(x, z)) {
            let _ = self.block_at(x, Y_MIN, z);
        }
        let top = self.surface_top(x, z).clamp(Y_MIN, Y_MAX);
        for y in (Y_MIN..=top).rev() {
            let block = self.block_at(x, y, z);
            if block != Block::Air && !block.is_tree() {
                return Some((y, block));
            }
        }
        None
    }

    /// 返回生成器在给定半开区域内部或树冠可能覆盖到该区域的树方块。
    ///
    /// 这个查询只依赖确定性高度图和坐标 hash，不需要预生成区域内的所有柱，
    /// 因此适合 LOD 分块按需构建远距离树木。
    pub fn tree_blocks_in_region(
        &self,
        x0: i64,
        x1: i64,
        z0: i64,
        z1: i64,
    ) -> Vec<(i64, i64, i64, Block)> {
        let treegen = TreeDecorator::new(self.column_gen.seed);
        let mut blocks: HashMap<(i64, i64, i64), Block> = HashMap::new();
        for tree in treegen.generate(&self.heightmap, x0, x1, z0, z1) {
            for (x, y, z, block) in tree.blocks() {
                blocks.insert((x, y, z), block);
            }
        }

        // 两棵相邻树的树冠可能重叠；LOD 直接按树提交几何时必须去重，
        // 否则同一个叶方块会被画两遍，产生 z-fighting / 深色接缝。
        let mut blocks: Vec<_> = blocks
            .into_iter()
            .map(|((x, y, z), block)| (x, y, z, block))
            .collect();
        blocks.sort_unstable_by_key(|(x, y, z, _)| (*x, *y, *z));
        blocks
    }

    pub fn edits(&self) -> impl Iterator<Item = (&(i64, i64, i64), &Block)> {
        self.edits.iter()
    }

    /// 恢复存档中的方块编辑; 同一位置多条编辑时后写覆盖先写,
    /// 与运行期 `set_block` 的语义一致.
    pub fn apply_edits(&mut self, edits: impl IntoIterator<Item = (i64, i64, i64, Block)>) {
        for (x, y, z, block) in edits {
            self.set_block_raw(x, y, z, block);
        }
    }

    /// Replace a mesher's edit snapshot, including edits removed since the last request.
    pub(crate) fn replace_edits(
        &mut self,
        edits: impl IntoIterator<Item = (i64, i64, i64, Block)>,
    ) {
        self.edits.clear();
        self.edit_column_tops.clear();
        self.apply_edits(edits);
    }

    /// 推进一帧方块重力。
    ///
    /// 当前只模拟已缓存列和显式编辑中的落体方块。每个候选方块会落到
    /// 下方连续空气的最低位置；同一列按 y 从高到低处理，因此一摞沙子
    /// 会自然地逐格跟随下落，不会在同一个 tick 内互相穿过。
    ///
    /// 返回本次是否有方块移动，调用方据此请求网格重建。
    pub fn tick_gravity(&mut self) -> bool {
        let mut candidates = Vec::new();

        // 生成列里的沙子/沙砾不是 edits 的一部分，因此也要从缓存列中
        // 收集。显式 Air 编辑会在真正处理候选时通过 block_at 覆盖它。
        for (&(x, z), column) in &self.columns {
            for (index, &block) in column.blocks.iter().enumerate() {
                if block.def().gravity {
                    candidates.push((x, Y_MIN + index as i64, z));
                }
            }
        }

        // 新放置的落体方块可能位于尚未生成/缓存的列中。
        candidates.extend(
            self.edits
                .iter()
                .filter_map(|(&(x, y, z), &block)| block.def().gravity.then_some((x, y, z))),
        );

        // HashMap 的遍历顺序不稳定；固定顺序既保证结果可复现，也保证
        // 同一列先处理高处方块。
        candidates.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)).then(b.1.cmp(&a.1)));
        candidates.dedup();

        let mut changed = false;
        for (x, y, z) in candidates {
            let block = self.block_at(x, y, z);
            if !block.def().gravity || y <= Y_MIN {
                continue;
            }

            let mut landing_y = y - 1;
            while landing_y >= Y_MIN && self.block_at(x, landing_y, z) == Block::Air {
                landing_y -= 1;
            }
            let destination_y = landing_y + 1;

            if destination_y < y {
                self.set_block_raw(x, y, z, Block::Air);
                self.set_block_raw(x, destination_y, z, block);
                changed = true;
            }
        }

        changed
    }

    /// Remove a block and refill it from an adjacent water block when one is
    /// present. The plain `VoxelWorld::set_block` implementation intentionally
    /// remains a raw edit because the mesh worker replays edits in arbitrary
    /// hash-map order and must not run simulation while doing so.
    pub fn remove_block_with_fluid_update(&mut self, x: i64, y: i64, z: i64) {
        self.set_block_raw(x, y, z, Block::Air);

        if !self.has_adjacent_water(x, y, z) {
            return;
        }

        self.set_block_raw(x, y, z, Block::Water);
    }

    fn has_adjacent_water(&mut self, x: i64, y: i64, z: i64) -> bool {
        const NEIGHBORS: [(i64, i64, i64); 6] = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ];

        NEIGHBORS
            .iter()
            .any(|&(dx, dy, dz)| self.block_at(x + dx, y + dy, z + dz) == Block::Water)
    }

    fn set_block_raw(&mut self, x: i64, y: i64, z: i64, block: Block) {
        if (Y_MIN..=Y_MAX).contains(&y) {
            self.edits.insert((x, y, z), block);
            self.edit_column_tops
                .entry((x, z))
                .and_modify(|top| *top = (*top).max(y))
                .or_insert(y);
        }
    }

    /// 返回该列可能包含方块的最高 y。
    ///
    /// 陆地需要扫描到地表，海洋还要扫描到水面；更高的位置一定是空气，
    /// 网格生成器可以据此跳过大量无意义的查询。
    pub fn surface_top(&self, x: i64, z: i64) -> i64 {
        let generated_top = self
            .heightmap
            .sample(x, z)
            .height
            .floor()
            .clamp(Y_MIN as f64, Y_MAX as f64)
            .max(SEA_Y as f64) as i64;
        let edited_top = self.edit_column_tops.get(&(x, z)).copied().unwrap_or(Y_MIN);
        let cached_top = self
            .columns
            .get(&(x, z))
            .map_or(Y_MIN, |column| Y_MIN + column.blocks.len() as i64 - 1);
        generated_top.max(edited_top).max(cached_top)
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

        let terrain = self.heightmap.sample(x, z);
        let column = self.column_gen.generate_with_terrain(x, z, terrain);
        let treegen = TreeDecorator::new(self.column_gen.seed);
        let column = decorate_column(column, &self.heightmap, &treegen);
        self.columns.insert((x, z), column);
        self.columns
            .get(&(x, z))
            .expect("刚插入的体素柱必须存在")
            .get(y)
    }

    fn set_block(&mut self, x: i64, y: i64, z: i64, block: Block) {
        self.set_block_raw(x, y, z, block);
    }
}

/// 把区域内所有候选树的落点投影到当前列。
///
/// 树冠会跨越列边界，因此每列都查询周围两格候选；位置 hash 保证不同
/// 列得到完全相同的结果，且只向空气中写装饰方块，不覆盖地形和水。
fn decorate_column(
    mut column: Column,
    heightmap: &WorldHeightmap,
    treegen: &TreeDecorator,
) -> Column {
    let x = column.x;
    let z = column.z;
    let trees: Vec<Tree> = treegen.generate(heightmap, x - 2, x + 3, z - 2, z + 3);
    for tree in trees {
        for (block_x, y, block_z, block) in tree.blocks() {
            if block_x != x || block_z != z || !(Y_MIN..=Y_MAX).contains(&y) {
                continue;
            }
            if column.get(y) != Block::Air {
                continue;
            }
            let required_len = (y - Y_MIN + 1) as usize;
            if column.blocks.len() < required_len {
                column.blocks.resize(required_len, Block::Air);
            }
            column.blocks[(y - Y_MIN) as usize] = block;
        }
    }
    column
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
    fn terrain_surface_block_skips_manual_tree_blocks() {
        let mut world = GeneratedVoxelWorld::new(1);
        let tree = Tree {
            x: 0,
            z: 0,
            ground: 100,
            height: 4,
        };
        for (x, y, z, block) in tree.blocks() {
            world.set_block(x, y, z, block);
        }

        let (leaf_x, leaf_y, leaf_z, _) = tree
            .blocks()
            .into_iter()
            .find(|(_, _, _, block)| *block == Block::OakLeaves)
            .expect("tree should have leaves");
        assert_eq!(world.block_at(leaf_x, leaf_y, leaf_z), Block::OakLeaves);

        let (surface_y, surface) = world
            .terrain_surface_block(leaf_x, leaf_z)
            .expect("terrain surface below tree");
        assert!(!surface.is_tree());
        assert!(surface_y < leaf_y);

        let (tree_top_y, tree_top) = world
            .surface_block(leaf_x, leaf_z)
            .expect("tree column has blocks");
        assert!(tree_top.is_tree());
        assert!(tree_top_y >= leaf_y);
    }

    #[test]
    fn tree_blocks_in_region_returns_generated_trees() {
        let seed = 2026_0904;
        let heightmap = WorldHeightmap::new(seed);
        let tree = TreeDecorator::new(seed)
            .generate(&heightmap, 400, 500, 900, 1000)
            .into_iter()
            .next()
            .expect("seed should generate at least one tree in the sampled region");
        let world = GeneratedVoxelWorld::new(seed);
        let blocks = world.tree_blocks_in_region(tree.x - 2, tree.x + 3, tree.z - 2, tree.z + 3);

        assert!(
            blocks
                .iter()
                .any(|(_, _, _, block)| *block == Block::OakLog)
        );
        assert!(
            blocks
                .iter()
                .any(|(_, _, _, block)| *block == Block::OakLeaves)
        );
    }

    #[test]
    fn tree_blocks_in_region_are_deduplicated() {
        let world = GeneratedVoxelWorld::new(2026_0904);
        let blocks = world.tree_blocks_in_region(400, 500, 900, 1000);
        assert!(!blocks.is_empty());

        let mut positions: Vec<_> = blocks.iter().map(|(x, y, z, _)| (*x, *y, *z)).collect();
        let before = positions.len();
        positions.sort_unstable();
        positions.dedup();
        assert_eq!(before, positions.len(), "相邻树冠重叠时产生了重复方块");
    }

    #[test]
    fn column_decoration_matches_global_tree_blocks() {
        let seed = 2026_0904;
        let heightmap = WorldHeightmap::new(seed);
        let treegen = TreeDecorator::new(seed);
        let tree = treegen
            .generate(&heightmap, 400, 500, 900, 1000)
            .into_iter()
            .next()
            .expect("测试区域应至少有一棵树");
        let mut world = GeneratedVoxelWorld::new(seed);
        for (x, y, z, block) in tree.blocks() {
            assert_eq!(
                world.block_at(x, y, z),
                block,
                "列装饰没有复现全局树 ({x},{y},{z})"
            );
        }
    }

    #[test]
    fn retaining_columns_evicts_distant_generations() {
        let mut world = GeneratedVoxelWorld::new(11);
        world.preload_columns_parallel(0, 4, 0, 4);
        assert_eq!(world.cached_column_count(), 25);

        world.retain_columns_within(0, 0, 2);
        assert_eq!(world.cached_column_count(), 9);

        // Evicted columns are regenerated deterministically on demand.
        let expected = GeneratedVoxelWorld::new(11).block_at(4, 70, 4);
        assert_eq!(world.block_at(4, 70, 4), expected);
        assert_eq!(world.cached_column_count(), 10);
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

    #[test]
    fn applying_saved_edits_overrides_generation_and_tracks_column_tops() {
        let mut world = GeneratedVoxelWorld::new(7);
        let original = world.block_at(2, 60, 2);
        world.apply_edits([
            (2, 60, 2, Block::DiamondOre),
            (2, 60, 2, Block::Air),
            (2, 70, 2, Block::Air),
        ]);
        // 后写覆盖先写, 最终是挖空.
        assert_eq!(world.block_at(2, 60, 2), Block::Air);
        assert_ne!(original, Block::Air);
        // 编辑顶要参与网格范围计算, 不能因为是 Air 就丢掉.
        assert!(world.surface_top(2, 2) >= 70);
    }

    #[test]
    fn removing_block_refills_from_adjacent_water() {
        let mut world = GeneratedVoxelWorld::new(7);
        world.set_block(0, 60, 0, Block::Water);
        world.set_block(1, 60, 0, Block::OakPlanks);

        world.remove_block_with_fluid_update(1, 60, 0);

        assert_eq!(world.block_at(1, 60, 0), Block::Water);
    }

    #[test]
    fn removing_block_without_adjacent_water_stays_air() {
        let mut world = GeneratedVoxelWorld::new(7);
        for (dx, dy, dz) in [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ] {
            world.set_block(1 + dx, 60 + dy, dz, Block::Stone);
        }
        world.set_block(1, 60, 0, Block::OakPlanks);

        world.remove_block_with_fluid_update(1, 60, 0);

        assert_eq!(world.block_at(1, 60, 0), Block::Air);
    }

    #[test]
    fn gravity_block_falls_to_lowest_air_above_support() {
        let mut world = GeneratedVoxelWorld::new(7);
        world.set_block(0, 5, 0, Block::Stone);
        for y in 6..20 {
            world.set_block(0, y, 0, Block::Air);
        }
        world.set_block(0, 20, 0, Block::Sand);

        assert!(world.tick_gravity());
        assert_eq!(world.block_at(0, 20, 0), Block::Air);
        assert_eq!(world.block_at(0, 6, 0), Block::Sand);
    }

    #[test]
    fn stacked_gravity_blocks_fall_in_order() {
        let mut world = GeneratedVoxelWorld::new(7);
        world.set_block(0, 5, 0, Block::Stone);
        for y in 6..10 {
            world.set_block(0, y, 0, Block::Air);
        }
        world.set_block(0, 10, 0, Block::Sand);
        world.set_block(0, 11, 0, Block::Gravel);

        assert!(world.tick_gravity());
        assert_eq!(world.block_at(0, 11, 0), Block::Gravel);
        assert_eq!(world.block_at(0, 6, 0), Block::Sand);

        assert!(world.tick_gravity());
        assert_eq!(world.block_at(0, 7, 0), Block::Gravel);
        assert_eq!(world.block_at(0, 6, 0), Block::Sand);
    }
}
