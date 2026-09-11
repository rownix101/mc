//! JEI 使用的轻量配方注册表。
//!
//! 配方和方块注册表一样保持稳定、可枚举，UI 只依赖方块 id，不把配方
//! 逻辑散落在 egui 绘制代码中。后续切换到 JSON 数据驱动时，可以直接
//! 用加载后的切片替换这里的内置表。

use super::block::Block;

pub const CRAFTING_SLOTS: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecipeKind {
    Crafting,
    Smelting,
}

impl RecipeKind {
    pub const fn display(self) -> &'static str {
        match self {
            Self::Crafting => "合成",
            Self::Smelting => "烧炼",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Recipe {
    pub id: &'static str,
    pub kind: RecipeKind,
    /// 有序的 3x3 合成格；烧炼配方只使用第一个格子。
    pub ingredients: [Option<Block>; CRAFTING_SLOTS],
    pub output: Block,
    pub output_count: u16,
}

impl Recipe {
    pub const fn crafting(
        id: &'static str,
        ingredients: [Option<Block>; CRAFTING_SLOTS],
        output: Block,
        output_count: u16,
    ) -> Self {
        Self {
            id,
            kind: RecipeKind::Crafting,
            ingredients,
            output,
            output_count,
        }
    }

    pub const fn smelting(
        id: &'static str,
        ingredient: Block,
        output: Block,
        output_count: u16,
    ) -> Self {
        Self {
            id,
            kind: RecipeKind::Smelting,
            ingredients: [
                Some(ingredient),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            output,
            output_count,
        }
    }
}

const fn one(block: Block) -> [Option<Block>; CRAFTING_SLOTS] {
    [Some(block), None, None, None, None, None, None, None, None]
}

const fn four(block: Block) -> [Option<Block>; CRAFTING_SLOTS] {
    [
        Some(block),
        Some(block),
        None,
        Some(block),
        Some(block),
        None,
        None,
        None,
        None,
    ]
}

/// 内置配方。数量少而明确，方便 JEI 在没有完整生存系统时仍然能展示
/// 合成与烧炼关系；新增方块时在这里追加配方即可。
pub const ALL: [Recipe; 4] = [
    Recipe::crafting("mc:oak_planks", one(Block::OakLog), Block::OakPlanks, 4),
    Recipe::crafting("mc:sandstone", four(Block::Sand), Block::Sandstone, 1),
    Recipe::smelting("mc:glass", Block::Sand, Block::Glass, 1),
    Recipe::smelting("mc:stone", Block::Cobblestone, Block::Stone, 1),
];

pub fn for_output(output: Block) -> impl Iterator<Item = &'static Recipe> {
    ALL.iter().filter(move |recipe| recipe.output == output)
}

pub fn using(ingredient: Block) -> impl Iterator<Item = &'static Recipe> {
    ALL.iter()
        .filter(move |recipe| recipe.ingredients.contains(&Some(ingredient)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipes_have_valid_outputs_and_ids() {
        let mut ids = std::collections::HashSet::new();
        for recipe in ALL {
            assert_ne!(recipe.output, Block::Air);
            assert!(recipe.output_count > 0);
            assert!(ids.insert(recipe.id));
            assert!(
                recipe
                    .ingredients
                    .iter()
                    .flatten()
                    .all(|block| *block != Block::Air)
            );
        }
    }

    #[test]
    fn lookup_supports_recipe_and_usage_views() {
        assert_eq!(for_output(Block::OakPlanks).count(), 1);
        assert_eq!(for_output(Block::Dirt).count(), 0);
        assert_eq!(using(Block::Sand).count(), 2);
        assert_eq!(using(Block::OakLog).count(), 1);
    }
}
