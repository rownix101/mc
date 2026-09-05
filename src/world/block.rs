//! 方块注册表: id / 属性 / 贴图映射.
//!
//! - id 稳定 (`#[repr(u16)]` 显式判别值). 存档/网络直接存 `u16`, 已分配 id 永不复用.
//! - `Air = 0` 保留为空气.
//! - 贴图全部 16x16, 来自 REFI (CC BY-SA 4.0, 见 `assets/textures/ATTRIBUTION.md`).
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
}

/// 方块总数 (含空气).
pub const COUNT: usize = 20;

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
        top: "default_grass.png",
        side: "default_grass_side.png",
        bottom: "default_dirt.png",
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
        top: "default_dirt.png",
        side: "default_dirt.png",
        bottom: "default_dirt.png",
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
        top: "default_stone.png",
        side: "default_stone.png",
        bottom: "default_stone.png",
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
        top: "default_cobble.png",
        side: "default_cobble.png",
        bottom: "default_cobble.png",
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
        top: "default_sand.png",
        side: "default_sand.png",
        bottom: "default_sand.png",
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
        top: "mcl_core_sandstone_normal.png",
        side: "mcl_core_sandstone_normal.png",
        bottom: "mcl_core_sandstone_normal.png",
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
        top: "default_gravel.png",
        side: "default_gravel.png",
        bottom: "default_gravel.png",
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
        top: "default_clay.png",
        side: "default_clay.png",
        bottom: "default_clay.png",
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
        top: "default_tree_top.png",
        side: "default_tree.png",
        bottom: "default_tree_top.png",
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
        top: "default_wood.png",
        side: "default_wood.png",
        bottom: "default_wood.png",
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
        top: "default_leaves.png",
        side: "default_leaves.png",
        bottom: "default_leaves.png",
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
        top: "default_glass_detail.png",
        side: "default_glass_detail.png",
        bottom: "default_glass_detail.png",
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
        top: "mcl_core_bedrock.png",
        side: "mcl_core_bedrock.png",
        bottom: "mcl_core_bedrock.png",
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
        top: "default_obsidian.png",
        side: "default_obsidian.png",
        bottom: "default_obsidian.png",
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
        top: "mcl_core_coal_ore.png",
        side: "mcl_core_coal_ore.png",
        bottom: "mcl_core_coal_ore.png",
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
        top: "mcl_core_iron_ore.png",
        side: "mcl_core_iron_ore.png",
        bottom: "mcl_core_iron_ore.png",
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
        top: "mcl_core_gold_ore.png",
        side: "mcl_core_gold_ore.png",
        bottom: "mcl_core_gold_ore.png",
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
        top: "mcl_core_diamond_ore.png",
        side: "mcl_core_diamond_ore.png",
        bottom: "mcl_core_diamond_ore.png",
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
        top: "default_water.png",
        side: "default_water.png",
        bottom: "default_water.png",
        render: RenderClass::Fluid,
        solid: false,
        fluid: true,
        gravity: false,
        hardness: None,
        blast: 100.0,
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
        assert_eq!(tex.len(), 21, "去重贴图数: {tex:?}");
        assert_eq!(tex[0], "default_grass.png");
        assert_eq!(tex[1], "default_grass_side.png");
        assert_eq!(tex[2], "default_dirt.png");
        assert_eq!(*tex.last().unwrap(), "default_water.png");
        // 原木顶底与侧面不同, 草方块三面模型正确.
        assert_eq!(Block::OakLog.tile(Face::Top), "default_tree_top.png");
        assert_eq!(Block::OakLog.tile(Face::Side), "default_tree.png");
        assert_eq!(Block::GrassBlock.tile(Face::Bottom), "default_dirt.png");
    }

    #[test]
    fn opacity_and_fluid_flags() {
        assert!(!Block::Air.is_opaque() && !Block::Air.is_solid());
        assert!(!Block::OakLeaves.is_opaque() && Block::OakLeaves.is_solid());
        assert!(!Block::Glass.is_opaque() && Block::Glass.is_solid());
        assert!(Block::Water.is_fluid() && !Block::Water.is_solid());
        assert!(Block::Water.is_transparent() && Block::Glass.is_transparent());
        assert!(!Block::OakLeaves.is_transparent() && !Block::Stone.is_transparent());
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
