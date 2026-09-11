//! 方块注册表: id / 属性 / 贴图映射.
//!
//! - id 稳定 (`#[repr(u16)]` 显式判别值). 存档/网络直接存 `u16`, 已分配 id 永不复用.
//! - `Air = 0` 保留为空气.
//! - 基础贴图全部 16x16，由 Poly Haven CC0 材质派生（来源和处理方式见
//!   `assets/textures/ATTRIBUTION.md`).
//! - 面模型: `Top / Bottom / Side` (四侧共用). 只有原木类 Top 与 Side 不同.
//! - JSON 数据驱动是 M8 的事 (见 TECH_ROADMAP 3.9); 注册表 API 已按"代码只认 id"设计,
//!   到时把 `REGISTRY` 换成 JSON 加载即可, 调用方不用改.

/// 方块 id 类型. 存档/网络/调色板里存这个.
pub type BlockId = u16;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u16)]
pub enum Block {
    Air = 0,
    GrassBlock = 1,
    Dirt = 2,
    Stone = 3,
    Cobblestone = 4,
    Sand = 5,
    Sandstone = 6,
    Gravel = 7,
    Clay = 8,
    OakLog = 9,
    OakPlanks = 10,
    OakLeaves = 11,
    Glass = 12,
    Bedrock = 13,
    Obsidian = 14,
    CoalOre = 15,
    IronOre = 16,
    GoldOre = 17,
    DiamondOre = 18,
    Water = 19,
    Bookshelf = 20,
    Beehive = 21,
    Furnace = 22,
    Piston = 23,
    RedstoneOre = 24,
    DeepslateRedstoneOre = 25,
}

/// 方块总数 (含空气).
pub const COUNT: usize = 26;

/// 光照强度上限 (与 Minecraft 一致: 0..=15).
pub const LIGHT_MAX: u8 = 15;
/// 不透明方块的光照衰减, 大于 [`LIGHT_MAX`] 即表示完全不透光.
pub const LIGHT_OPAQUE: u8 = LIGHT_MAX + 1;

/// 按 id 升序的全部方块. id 即数组下标, 改表时必须同步.
pub const ALL: [Block; COUNT] = [
    Block::Air,
    Block::GrassBlock,
    Block::Dirt,
    Block::Stone,
    Block::Cobblestone,
    Block::Sand,
    Block::Sandstone,
    Block::Gravel,
    Block::Clay,
    Block::OakLog,
    Block::OakPlanks,
    Block::OakLeaves,
    Block::Glass,
    Block::Bedrock,
    Block::Obsidian,
    Block::CoalOre,
    Block::IronOre,
    Block::GoldOre,
    Block::DiamondOre,
    Block::Water,
    Block::Bookshelf,
    Block::Beehive,
    Block::Furnace,
    Block::Piston,
    Block::RedstoneOre,
    Block::DeepslateRedstoneOre,
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Face {
    Top,
    Bottom,
    Side,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderClass {
    /// 空气: 不渲染、不参与 cull.
    Hidden,
    /// 完全不透明: 可剔除相邻面.
    Opaque,
    /// 树叶: cutout, 自身不剔除邻面, 同种相邻剔除.
    Cutout,
    /// 玻璃: 透明, 同种相邻剔除.
    Transparent,
    /// 水: 半透明流体, 同种相邻剔除.
    Fluid,
}

pub struct BlockDef {
    pub id: BlockId,
    /// 注册名, 代码只认这个 (JSON 化后不变).
    pub name: &'static str,
    pub display: &'static str,
    /// 贴图文件名 (`assets/textures/` 下).
    pub top: &'static str,
    pub side: &'static str,
    pub bottom: &'static str,
    pub render: RenderClass,
    /// 碰撞/可站立.
    pub solid: bool,
    pub fluid: bool,
    /// 沙砾类落体标记 (tick 系统用, 此处只标记不模拟).
    pub gravity: bool,
    /// 挖掘硬度. `None` = 不可挖掘 (基岩/空气/水).
    pub hardness: Option<f32>,
    pub blast: f32,
}

// NOTE: 加新方块时往后追加 id, 并同步 ALL / REGISTRY / from_id.
const REGISTRY: [BlockDef; COUNT] = [
    BlockDef {
        id: 0,
        name: "mc:air",
        display: "空气",
        top: "",
        side: "",
        bottom: "",
        render: RenderClass::Hidden,
        solid: false,
        fluid: false,
        gravity: false,
        hardness: None,
        blast: 0.0,
    },
    BlockDef {
        id: 1,
        name: "mc:grass_block",
        display: "草方块",
        top: "grass_block_top.png",
        side: "grass_block_side_overlay.png",
        bottom: "dirt.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(0.6),
        blast: 0.6,
    },
    BlockDef {
        id: 2,
        name: "mc:dirt",
        display: "泥土",
        top: "dirt.png",
        side: "dirt.png",
        bottom: "dirt.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(0.5),
        blast: 0.5,
    },
    BlockDef {
        id: 3,
        name: "mc:stone",
        display: "石头",
        top: "stone.png",
        side: "stone.png",
        bottom: "stone.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(1.5),
        blast: 6.0,
    },
    BlockDef {
        id: 4,
        name: "mc:cobblestone",
        display: "圆石",
        top: "cobblestone.png",
        side: "cobblestone.png",
        bottom: "cobblestone.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(2.0),
        blast: 6.0,
    },
    BlockDef {
        id: 5,
        name: "mc:sand",
        display: "沙子",
        top: "sand.png",
        side: "sand.png",
        bottom: "sand.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: true,
        hardness: Some(0.5),
        blast: 0.5,
    },
    BlockDef {
        id: 6,
        name: "mc:sandstone",
        display: "砂岩",
        top: "sandstone_top.png",
        side: "sandstone.png",
        bottom: "sandstone_bottom.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(0.8),
        blast: 0.8,
    },
    BlockDef {
        id: 7,
        name: "mc:gravel",
        display: "沙砾",
        top: "gravel.png",
        side: "gravel.png",
        bottom: "gravel.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: true,
        hardness: Some(0.6),
        blast: 0.6,
    },
    BlockDef {
        id: 8,
        name: "mc:clay",
        display: "黏土块",
        top: "clay.png",
        side: "clay.png",
        bottom: "clay.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(0.6),
        blast: 0.6,
    },
    BlockDef {
        id: 9,
        name: "mc:oak_log",
        display: "橡木原木",
        top: "oak_log_top.png",
        side: "oak_log.png",
        bottom: "oak_log_top.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(2.0),
        blast: 2.0,
    },
    BlockDef {
        id: 10,
        name: "mc:oak_planks",
        display: "橡木板",
        top: "oak_planks.png",
        side: "oak_planks.png",
        bottom: "oak_planks.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(2.0),
        blast: 3.0,
    },
    BlockDef {
        id: 11,
        name: "mc:oak_leaves",
        display: "橡树叶",
        top: "oak_leaves.png",
        side: "oak_leaves.png",
        bottom: "oak_leaves.png",
        render: RenderClass::Cutout,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(0.2),
        blast: 0.2,
    },
    BlockDef {
        id: 12,
        name: "mc:glass",
        display: "玻璃",
        top: "glass.png",
        side: "glass.png",
        bottom: "glass.png",
        render: RenderClass::Transparent,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(0.3),
        blast: 0.3,
    },
    BlockDef {
        id: 13,
        name: "mc:bedrock",
        display: "基岩",
        top: "bedrock.png",
        side: "bedrock.png",
        bottom: "bedrock.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: None,
        blast: 3_600_000.0,
    },
    BlockDef {
        id: 14,
        name: "mc:obsidian",
        display: "黑曜石",
        top: "obsidian.png",
        side: "obsidian.png",
        bottom: "obsidian.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(50.0),
        blast: 1200.0,
    },
    BlockDef {
        id: 15,
        name: "mc:coal_ore",
        display: "煤矿",
        top: "coal_ore.png",
        side: "coal_ore.png",
        bottom: "coal_ore.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(3.0),
        blast: 3.0,
    },
    BlockDef {
        id: 16,
        name: "mc:iron_ore",
        display: "铁矿",
        top: "iron_ore.png",
        side: "iron_ore.png",
        bottom: "iron_ore.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(3.0),
        blast: 3.0,
    },
    BlockDef {
        id: 17,
        name: "mc:gold_ore",
        display: "金矿",
        top: "gold_ore.png",
        side: "gold_ore.png",
        bottom: "gold_ore.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(3.0),
        blast: 3.0,
    },
    BlockDef {
        id: 18,
        name: "mc:diamond_ore",
        display: "钻石矿",
        top: "diamond_ore.png",
        side: "diamond_ore.png",
        bottom: "diamond_ore.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(3.0),
        blast: 3.0,
    },
    BlockDef {
        id: 19,
        name: "mc:water",
        display: "水",
        top: "water_overlay.png",
        side: "water_overlay.png",
        bottom: "water_overlay.png",
        render: RenderClass::Fluid,
        solid: false,
        fluid: true,
        gravity: false,
        hardness: None,
        blast: 100.0,
    },
    BlockDef {
        id: 20,
        name: "mc:bookshelf",
        display: "书架",
        top: "bookshelf_top.png",
        side: "bookshelf.png",
        bottom: "bookshelf_top.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(1.5),
        blast: 1.5,
    },
    BlockDef {
        id: 21,
        name: "mc:beehive",
        display: "蜂箱",
        top: "beehive_end.png",
        side: "beehive_side.png",
        bottom: "beehive_end.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(0.6),
        blast: 0.6,
    },
    BlockDef {
        id: 22,
        name: "mc:furnace",
        display: "熔炉",
        top: "furnace_top.png",
        side: "furnace_side.png",
        bottom: "furnace_top.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(3.5),
        blast: 3.5,
    },
    BlockDef {
        id: 23,
        name: "mc:piston",
        display: "活塞",
        top: "piston_top.png",
        side: "piston_side.png",
        bottom: "piston_inner.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(1.5),
        blast: 1.5,
    },
    BlockDef {
        id: 24,
        name: "mc:redstone_ore",
        display: "红石矿",
        top: "redstone_ore.png",
        side: "redstone_ore.png",
        bottom: "redstone_ore.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(3.0),
        blast: 3.0,
    },
    BlockDef {
        id: 25,
        name: "mc:deepslate_redstone_ore",
        display: "深层红石矿",
        // Pixel Perfection CE targets Minecraft 1.16, before deepslate existed.
        top: "redstone_ore.png",
        side: "redstone_ore.png",
        bottom: "redstone_ore.png",
        render: RenderClass::Opaque,
        solid: true,
        fluid: false,
        gravity: false,
        hardness: Some(4.5),
        blast: 3.0,
    },
];

impl Block {
    pub fn id(self) -> BlockId {
        self as BlockId
    }

    pub fn def(self) -> &'static BlockDef {
        &REGISTRY[self as usize]
    }

    pub fn from_id(id: BlockId) -> Option<Block> {
        ALL.get(id as usize).copied()
    }

    pub fn from_name(name: &str) -> Option<Block> {
        ALL.iter().copied().find(|b| b.def().name == name)
    }

    pub fn tile(self, face: Face) -> &'static str {
        let d = self.def();
        match face {
            Face::Top => d.top,
            Face::Bottom => d.bottom,
            Face::Side => d.side,
        }
    }

    pub fn is_opaque(self) -> bool {
        self.def().render == RenderClass::Opaque
    }

    pub fn is_solid(self) -> bool {
        self.def().solid
    }

    pub fn is_fluid(self) -> bool {
        self.def().fluid
    }

    pub fn is_transparent(self) -> bool {
        matches!(
            self.def().render,
            RenderClass::Transparent | RenderClass::Fluid
        )
    }

    /// 供体素着色器区分的渲染材质编码:
    /// `0` = 普通不透明/alpha-test, `1` = 流体, `2` = 树叶 cutout, `3` = 草方块, `4` = 连续混合沙地.
    ///
    /// 树叶单独编码后可以在片元着色器里做背面透光和参与柔和的丁达尔
    /// 屏幕空间散射; 它仍然写不透明深度、通过 atlas alpha 做镂空。
    pub fn render_material(self) -> f32 {
        if self == Self::Sand {
            return 4.0;
        }
        if self == Self::GrassBlock {
            return 3.0;
        }
        match self.def().render {
            RenderClass::Fluid => 1.0,
            RenderClass::Cutout => 2.0,
            _ => 0.0,
        }
    }

    /// 生成树木使用的方块（原木/树叶）。LOD 需要把它们和地形表面分开处理。
    pub fn is_tree(self) -> bool {
        matches!(self, Block::OakLog | Block::OakLeaves)
    }

    /// 光穿过一格该方块后的衰减. 不透明方块返回 [`LIGHT_OPAQUE`], 表示完全
    /// 不透光; 其它方块至少衰减 1, 水和树叶更多, 用来表现海深和树荫.
    pub fn light_attenuation(self) -> u8 {
        if self.is_opaque() {
            return LIGHT_OPAQUE;
        }
        match self {
            Block::Water => 2,
            Block::OakLeaves => 2,
            _ => 1,
        }
    }

    /// 方块自发光强度 (`0..=LIGHT_MAX`). 目前注册表里还没有发光方块; 接入
    /// 火把 / 萤石后在这里返回非零即可, [`crate::world::light`] 会自动拾取.
    pub fn light_emission(self) -> u8 {
        0
    }

    /// 去重后的全部贴图文件名, 顺序 = 按 id 遍历、每块按 Top/Side/Bottom 首次出现顺序.
    /// 该顺序即 atlas 的 TileId 分配顺序, 稳定不变.
    pub fn unique_textures() -> Vec<&'static str> {
        let mut out = Vec::new();
        for b in ALL {
            if b == Block::Air {
                continue;
            }
            for t in [b.def().top, b.def().side, b.def().bottom] {
                if !out.contains(&t) {
                    out.push(t);
                }
            }
        }
        out
    }
}

/// 面剔除判定: `cur` 的朝向 `neighbor` 的那个面要不要画.
/// 规则: 空气不画; 邻居是空气则画; 邻居不透明则不画;
/// 同种透明/流体/cutout 相邻不画 (玻璃-玻璃、水-水、叶-叶); 其余画.
pub fn should_draw_face(cur: Block, neighbor: Block) -> bool {
    if cur == Block::Air {
        return false;
    }
    if neighbor == Block::Air {
        return true;
    }
    if neighbor.is_opaque() {
        return false;
    }
    if cur == neighbor {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_stable() {
        assert_eq!(Block::Air.id(), 0);
        assert_eq!(Block::GrassBlock.id(), 1);
        assert_eq!(Block::Water.id(), 19);
        assert_eq!(Block::Bookshelf.id(), 20);
        assert_eq!(Block::DeepslateRedstoneOre.id(), 25);
        for (i, b) in ALL.iter().enumerate() {
            assert_eq!(*b as usize, i, "ALL 顺序必须与判别值一致");
            assert_eq!(b.def().id as usize, i);
            assert_eq!(Block::from_id(i as u16), Some(*b));
        }
        assert_eq!(Block::from_id(COUNT as u16), None);
        assert_eq!(Block::from_name("mc:stone"), Some(Block::Stone));
        assert_eq!(Block::from_name("mc:nope"), None);
    }

    #[test]
    fn textures_complete() {
        let tex = Block::unique_textures();
        assert_eq!(tex.len(), 33, "去重贴图数: {tex:?}");
        assert_eq!(tex[0], "grass_block_top.png");
        assert_eq!(tex[1], "grass_block_side_overlay.png");
        assert_eq!(tex[2], "dirt.png");
        assert_eq!(tex[22], "water_overlay.png");
        assert_eq!(tex[31], "piston_inner.png");
        // 原木顶底与侧面不同, 草方块三面模型正确.
        assert_eq!(Block::OakLog.tile(Face::Top), "oak_log_top.png");
        assert_eq!(Block::OakLog.tile(Face::Side), "oak_log.png");
        assert_eq!(Block::GrassBlock.tile(Face::Bottom), "dirt.png");
    }

    #[test]
    fn opacity_and_fluid_flags() {
        assert!(!Block::Air.is_opaque() && !Block::Air.is_solid());
        assert!(!Block::OakLeaves.is_opaque() && Block::OakLeaves.is_solid());
        assert!(!Block::Glass.is_opaque() && Block::Glass.is_solid());
        assert!(Block::Water.is_fluid() && !Block::Water.is_solid());
        assert!(Block::Water.is_transparent() && Block::Glass.is_transparent());
        assert!(!Block::OakLeaves.is_transparent() && !Block::Stone.is_transparent());
        // Shader material codes: ordinary = 0, water = 1, cutout leaves = 2, grass = 3.
        assert_eq!(Block::Stone.render_material(), 0.0);
        assert_eq!(Block::GrassBlock.render_material(), 3.0);
        assert_eq!(Block::Water.render_material(), 1.0);
        assert_eq!(Block::OakLeaves.render_material(), 2.0);
        assert!(Block::Sand.def().gravity && Block::Gravel.def().gravity);
        assert!(!Block::Stone.def().gravity);
        assert_eq!(Block::Bedrock.def().hardness, None);
    }

    #[test]
    fn face_culling_rules() {
        // 石头藏在石头里不画, 露在空气中画.
        assert!(!should_draw_face(Block::Stone, Block::Stone));
        assert!(!should_draw_face(Block::Stone, Block::Dirt));
        assert!(should_draw_face(Block::Stone, Block::Air));
        // 石头贴着玻璃/水/叶, 石头面要画 (邻居透明).
        assert!(should_draw_face(Block::Stone, Block::Glass));
        assert!(should_draw_face(Block::Stone, Block::Water));
        // 玻璃贴石头不画; 玻璃贴玻璃不画; 玻璃贴空气画.
        assert!(!should_draw_face(Block::Glass, Block::Stone));
        assert!(!should_draw_face(Block::Glass, Block::Glass));
        assert!(should_draw_face(Block::Glass, Block::Air));
        // 水同理; 空气永远不画.
        assert!(!should_draw_face(Block::Water, Block::Water));
        assert!(should_draw_face(Block::Water, Block::Air));
        assert!(!should_draw_face(Block::Air, Block::Stone));
    }
}
