# 类 MC 游戏 · 技术路线 (Tech Roadmap)

> 状态：已确立 / 锁定。无引擎，`Rust + wgpu` 自研。
> 更新：2026-09-04。视距：半径 64 chunks。脚本：`rhai`。

## 1. 目标 (相比原版 MC)

1. **世界规模**：更大视距 (64 chunks 半径)、超远距离坐标稳定、更快的 chunk streaming。
2. **地形**：更强的大尺度地理结构，而非主要靠噪声拼接。
3. **渲染**：GPU-driven、indirect draw、Hi-Z / occlusion、better LOD、compute meshing。
4. **光照**：更好的全局光照、体积雾、软阴影、动态天空。
5. **世界模拟**：更丰富的流体、温度、生态、侵蚀、材料属性。
6. **红石 / 自动化**：更统一、更高性能的逻辑网络。
7. **Modding**：数据驱动甚至脚本化，而非大量硬编码。

非目标 (M1–M3 不做)：生存玩法、联机、手机 / wasm、Mesh Shader 主路径、体素 GI。

## 2. 技术栈 (锁定版本)

| 用途 | 选型 | 版本 / 说明 |
|---|---|---|
| 图形 | `wgpu` (WGSL) | `30.0.1`，Vulkan / Metal / DX12 / GLES 抽象 |
| 窗口输入 | `winit` | `0.30.13` 稳定版，不用 `0.31-beta` |
| 数学 | `glam` | `0.30`，渲染侧 SIMD |
| Debug UI | `egui` + `egui-wgpu` + `egui-winit` | `0.36.1`，仅 F3 / 菜单，feature-gated |
| 异步初始化 | `pollster` | `0.4`，阻塞创建 Device (wgpu 官方示例用法) |
| 噪声 | `fastnoise-lite` | `1.1`，只做细节，宏观不用它 |
| 并行 | `rayon` + `crossbeam-channel` + `parking_lot` | chunk 生成 / meshing 后台线程池 + 无锁队列 |
| 顶点零拷贝 | `bytemuck` | `1.23` + `derive` |
| 纹理加载 | `image` | `0.25`，只做 atlas 加载 |
| 存档 | `serde` + `postcard` + `lz4` | Region 二进制 + 压缩，热路径不用 JSON |
| 日志 /  profiling | `tracing` + `tracing-subscriber` | 后期加 `tracy-client` (feature-gated) |
| LOD / 压缩 / 索引 | `half` + `hashbrown` + `slotmap` + `bitvec` | `2.4` / `0.15` / `1.0` / `1.0`，64 视距必需 |
| 数据驱动 | `serde_json` + `toml` | registry / 配置 |
| 脚本 | `rhai` | `1.26.0`，只做注册表 / 配方 / 事件 / 逻辑回调 |

工具链：`Rust 1.87+, edition 2024`，单 binary crate 起步，桌面 only (Windows / Linux)。

明确不用：

- 不用 `bevy` / `hecs` ECS，前期纯 data + 函数。
- 不用 `rapier` / `physx`，体素只需手写 AABB + DDA。
- 不用 `raw OpenGL` / `glfw`，跨平台成本高。
- 不用 `wgpu` Mesh Shader 做主路径，可移植性差。

起手 `Cargo.toml`：

```toml
[dependencies]
wgpu = "30.0.1"
winit = "0.30.13"
glam = "0.30"
egui = "0.36.1"
egui-wgpu = "0.36"
egui-winit = "0.36"
pollster = "0.4"
fastnoise-lite = "1.1"
rayon = "1.10"
crossbeam-channel = "0.5"
parking_lot = "0.12"
bytemuck = { version = "1.23", features = ["derive"] }
image = "0.25"
serde = { version = "1", features = ["derive"] }
serde_json = "1.0"
toml = "0.8"
postcard = { version = "1", features = ["use-std"] }
lz4 = "1.10"
rhai = "1.26.0"
half = "2.4"
hashbrown = "0.15"
slotmap = "1.0"
bitvec = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
```

## 3. 核心架构决策 (已锁定)

### 3.1 坐标系：超远距离稳定

- 权威：`i64 block` + `f64 world`。
- 渲染：`f32 camera-relative` 进 shader。
- 每 4096 块 origin rebase 一次。
- `glam` 只管渲染侧，不做权威模拟。

### 3.2 Chunk 与 LOD 环 (64 视距)

- Chunk 尺寸：`32^3`，列式 `32 x H x 32` 管理。存储 `u16 block id + palette`。
- 64 chunks 半径 = 半径 2048 块，全精度常驻不可能，按环降级：

| 环 | 距离 (chunks) | 内容 |
|---|---|---|
| 近 | `0–10` | 全精度 voxel + AO + 碰撞 + tick |
| 中 | `10–24` | 简化网格 `step 2/4`，无碰撞 / 无 tick |
| 远 | `24–64` | 粗高度场 / 材料，不存全 voxel，走磁盘缓存 + 内存 cap 驱逐 |

### 3.3 流式 (streaming)

- `rayon` 线程池 + `crossbeam` 优先级队列，三队列：Near / Far / LOD。
- 每帧预算 (生成 / meshing / 上传分别限时)，nearest-first。
- `loaded cap + 显存 cap` 驱逐最远 chunk。
- 纯函数世界生成 + seed 可复现，方便测试和 LOD 复用。

### 3.4 渲染管线

- M1–M2：Forward 单 pass，`opaque + alpha-test + transparent(water)` 分开，每 chunk 1–2 draw call，texture atlas + 统一 uniform (camera / fog / sun)。
- 预留接口：`super-chunk 4x4 合并 + multi_draw_indirect`，不阻塞起步。
- M4 才做：GPU Hi-Z 金字塔 compute + occlusion culling、compute meshing、远 LOD 独立网格 (Transvoxel / decimated，不复用近网格)。
- `winit 0.30` 模型：`ApplicationHandler + resumed() 里建窗`。

### 3.5 地形：大尺度结构优先

管线定死，不许纯噪声拼接：

```
板块 / 大陆 mask → 气候 (温湿) → 水文 / 侵蚀 → 生物群系 → 洞穴 / 结构
```

- 宏观用低频 SDF + 模拟，`fastnoise-lite` 只做细节。
- WorldGen 必须是纯函数，便于 LOD / 测试 / 存档复用。

### 3.6 光照 / 雾 / 天空

- 动态天空：Preetham / Hillaire LUT，sun + fog 统一 uniform。
- 阴影：CSM 近景软阴影，远景只做高度雾 + aerial perspective。
- 先做 `flood-fill sky/block light + GTAO`，体积雾用半分辨率 raymarch 后叠加。
- 体素 GI / SDF GI 放最后 (M8 之后)。

### 3.7 世界模拟 / 流体 / 材料

- Material 属性表数据驱动：`density, viscosity, heat_capacity, fertility` 全放 JSON。
- Tick：`20 TPS 仿真 tick + chunk tick scheduler`，只 tick 近环 + 脏区。
- 流体用高度场 cellular automata 起步，温度 / 生态只存 coarse field，不逐 voxel。
- 侵蚀放世界生成期 + 低频后台模拟，不进主 tick。

### 3.8 红石 / 自动化 (逻辑网络)

- 不做 MC 式每 tick 全扫描。
- 统一成 `逻辑图 + 脏传播 + 分区 tick`，组件只有 `node / port / wire` 三种。
- 目标：上万门不卡。M7 才做，M1 预留 block tick 接口即可。

### 3.9 Modding / 脚本

- M1 就定：`block / item / biome / recipe = JSON + texture atlas`，代码只认 registry id。
- `rhai` 只做注册表 / 配方 / 世界事件 / 逻辑块回调，不进逐 voxel 热循环。
- `Engine::register_fn` 暴露有限 API (`set_block / get_biome` 等)。
- 高性能 mod 后期再上 `wasmtime (WASM)` 沙箱，不在当前路线内。

## 4. 工程结构

```
src/
  main.rs app.rs
  world/{block.rs chunk.rs worldgen.rs mesher.rs}
  render/{gpu.rs mesh.rs texture.rs shader.wgsl}
  player/{camera.rs physics.rs}
  sim/{tick.rs fluid.rs climate.rs}
  logic/{graph.rs}
  modding/{registry.rs scripting.rs}
content/
  blocks/*.json items/*.json recipes/*.json biomes/*.json
assets/
  textures/ shaders/
docs/
  TECH_ROADMAP.md  # 本文件
```

## 5. 里程碑

- **M1 Foundation**：空窗 + wgpu 清屏 + `ApplicationHandler` + egui overlay + 坐标 / registry 骨架。
- **M2 Near + Player**：palette chunk 存储、face-cull + AO meshing、atlas、AABB 物理 + DDA、可挖可放。
- **M3 Streaming + Macro terrain**：优先级队列 + 预算 + caps、大陆 / 气候 / 水文管线、磁盘缓存。
- **M4 LOD + GPU-driven**：简化网格、super-chunk indirect、Hi-Z / occlusion、compute meshing。
- **M5 Lighting / Sky**：CSM 软阴影、flood-fill 光照 + GTAO、动态天空 LUT、体积雾。
- **M6 Sim / Materials**：材料表、流体 CA、温湿 coarse field、侵蚀、20 TPS scheduler。
- **M7 Logic**：逻辑图 + 脏传播 + 分区 tick。
- **M8 Modding**：JSON 全量 + rhai API 稳定 + 热重载。
- **M9 Perf / Persist**：Region (`postcard` + `lz4`)、tracy、render-scale 安全阀、bench。

顺序原则：坐标 + 流式 > 近渲染跑通 > LOD / 地形宏观 > CSM / 雾 > 流体 / tick > 逻辑网络 > 脚本 / GI。

## 6. 风险

- 64 视距显存 / 内存爆炸 → 靠远环高度场 + 磁盘缓存 + caps 硬约束。
- `wgpu` 大版本迭代快 → 锁定 `30.0.1`，升级只在里程碑间隙做。
- compute meshing / Hi-Z 复杂度高 → 推迟到 M4，M1–M3 用 CPU 路径验证玩法。
