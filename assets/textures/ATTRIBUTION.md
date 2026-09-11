# 纹理来源 (ATTRIBUTION)

当前方块注册表使用由 [Poly Haven](https://polyhaven.com/textures) 摄影扫描材质
派生的 512×512 PBR 贴图。每个方块面包含 albedo、OpenGL normal 和
roughness；Poly Haven 资源采用 **CC0**：<https://polyhaven.com/license>。

## Poly Haven（当前使用）

| Poly Haven asset | 作者 | 本地用途 |
|---|---|---|
| [Leafy Grass](https://polyhaven.com/a/leafy_grass) | Charlotte Baglioni | 草地、草方块侧边过渡 |
| [Dirt Aerial 02](https://polyhaven.com/a/dirt_aerial_02) | Rob Tuytel | 泥土 |
| [Rock Ground 02](https://polyhaven.com/a/rock_ground_02) | Rob Tuytel | 石头、矿石基底、水色细节 |
| [Cobblestone 02](https://polyhaven.com/a/cobblestone_02) | Amal Kumar | 圆石、熔炉 |
| [Sand 03](https://polyhaven.com/a/sand_03) | Charlotte Baglioni | 沙、砂岩顶面 |
| [Sandstone Blocks 04](https://polyhaven.com/a/sandstone_blocks_04) | Rob Tuytel | 砂岩侧面/底面 |
| [Gravel Stones](https://polyhaven.com/a/gravel_stones) | Amal Kumar | 沙砾 |
| [Clay Plaster](https://polyhaven.com/a/clay_plaster) | Amal Kumar | 黏土 |
| [Bark Brown 02](https://polyhaven.com/a/bark_brown_02) | Rob Tuytel | 原木树皮 |
| [Brown Planks 04](https://polyhaven.com/a/brown_planks_04) | Rob Tuytel | 木板、书架、活塞木面 |
| [Forest Leaves 02](https://polyhaven.com/a/forest_leaves_02) | Rob Tuytel | 树叶 cutout |
| [Volcanic Rock Tiles](https://polyhaven.com/a/volcanic_rock_tiles) | Charlotte Baglioni | 基岩、黑曜石 |
| [Metal Plate 02](https://polyhaven.com/a/metal_plate_02) | Rob Tuytel | 玻璃细节、活塞金属面 |
| [Fine Grained Wood](https://polyhaven.com/a/fine_grained_wood) | Rob Tuytel | 原木端面、蜂箱 |

`tools/build_polyhaven_textures.py` 从 Poly Haven API 读取官方 1K JPG 与校验值，
再生成当前 33 组 512px 贴图。矿石脉、书架、蜂箱、熔炉和活塞的辨识细节由脚本
叠加在上表的 Poly Haven PBR 基底上；透明度遮罩也由脚本生成。原始 1K 文件仅
缓存在 `target/polyhaven-source/`，不进入仓库。

渲染时，自然材质按世界坐标连续取 2×2 子区域（每方块有效 256×256），草皮侧缘
使用带羽化的覆盖层；atlas 每格另有 8px 扩边并生成 mipmap，以减少相邻 tile 串色、
远景闪烁和逐方块重复感。

以下是目录中仍保留但不再由当前方块注册表引用的旧贴图来源说明。

## 基础材质：REFI

- 作者：MysticTempest
- 项目：REFI_Textures — <https://github.com/MysticTempest/REFI_Textures>
- 协议：**CC BY-SA 4.0** — <https://creativecommons.org/licenses/by-sa/4.0/>

文件映射（全部 16×16）：

| 本地文件 | REFI 原路径 (`textures/default_mcl_core/…`) |
|---|---|
| `default_grass.png` | `soil/default_grass.png` |
| `default_grass_side.png` | `soil/default_grass_side.png` |
| `default_dirt.png` | `soil/default_dirt.png` |
| `default_stone.png` | `stone/default_stone.png` |
| `default_cobble.png` | `stone/default_cobble.png` |
| `default_gravel.png` | `stone/default_gravel.png` |
| `mcl_core_bedrock.png` | `stone/mcl_core_bedrock.png` |
| `default_obsidian.png` | `stone/default_obsidian.png` |
| `default_sand.png` | `sand/default_sand.png` |
| `mcl_core_sandstone_normal.png` | `sand/mcl_core_sandstone_normal.png` |
| `default_tree.png` | `trees/default_tree.png`（原木侧面） |
| `default_tree_top.png` | `trees/default_tree_top.png`（原木顶底） |
| `default_wood.png` | `trees/default_wood.png`（木板） |
| `default_leaves.png` | `trees/default_leaves.png`（树叶） |
| `mcl_core_coal_ore.png` | `minerals_ores/mcl_core_coal_ore.png` |
| `mcl_core_iron_ore.png` | `minerals_ores/mcl_core_iron_ore.png` |
| `mcl_core_gold_ore.png` | `minerals_ores/mcl_core_gold_ore.png` |
| `mcl_core_diamond_ore.png` | `minerals_ores/mcl_core_diamond_ore.png` |
| `default_glass_detail.png` | `glass/default_glass_detail.png` |
| `default_clay.png` | `soil/default_clay.png` |
| `default_water.png` | `default_water.png`（静态水；动画版未用） |

合规说明：

1. 本目录 PNG 为原样 vendoring（仅改名存放），BY 归属见上表。
2. `target/atlas.png`（构建产物）属于改编作品（Adapted Material），同样按 CC BY-SA 4.0 分享。
3. 本仓库自有代码不受 CC BY-SA 约束；只有 `assets/textures/` 下原图及 `atlas.png` 这类衍生图适用 SA。

## 3D Default 兼容材质

- 作者/项目：GeForceLegend — <https://github.com/GeForceLegend/Minecraft-3D-Default>
- 来源版本：`1.21.2` 分支，提交 `bfc9ea0a7e75aa4e2d3beeb81fbb68d6f054ead5`
- 协议：**GNU GPL v3.0**，完整文本见 [`3d_default/LICENSE`](3d_default/LICENSE)

这些文件取自该资源包 `assets/minecraft/textures/block/`，仅选用了当前引擎
立方体 atlas 可直接使用的 16×16 彩色贴图：

| 本地文件 | 原路径 |
|---|---|
| `3d_default_bookshelf.png` | `assets/minecraft/textures/block/bookshelf.png` |
| `3d_default_beehive.png` | `assets/minecraft/textures/block/beehive_front.png` |
| `3d_default_furnace.png` | `assets/minecraft/textures/block/furnace_inside.png` |
| `3d_default_piston.png` | `assets/minecraft/textures/block/piston_arm.png` |
| `3d_default_redstone_ore.png` | `assets/minecraft/textures/block/redstone_ore_on.png` |
| `3d_default_deepslate_redstone_ore.png` | `assets/minecraft/textures/block/deepslate_redstone_ore_on.png` |

目标项目是基于原版资源的覆盖包，完整的 3D 模型和原版基础贴图不在其仓库中；
本项目当前是纯立方体渲染器，因此只接入了兼容的面贴图，未复制模型或灰度光照辅助图。
