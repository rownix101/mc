# 天空盒材质来源 (ATTRIBUTION)

以下为保留的历史天空素材，当前程序化天空已不再加载。素材来自 [Poly Haven](https://polyhaven.com/) 的 Clear Pure Sky HDRI：

| Poly Haven asset | 作者 | 本地用途 |
|---|---|---|
| [Kloofendal 43d Clear (Pure Sky)](https://polyhaven.com/a/kloofendal_43d_clear_puresky) | Greg Zaal | 静态天空底色、地平线与天顶颜色 |

`kloofendal_43d_clear_puresky_2k.png` 由官方 2K HDR 导出并做了以下离线处理：

- Reinhard 色调映射到 8-bit sRGB；
- 移除 HDRI 中已经存在的太阳，避免与运行时动态太阳重叠；
- 大范围天空渐变保留在贴图中；此贴图仅保留为历史素材。

Poly Haven 资源采用 **CC0**：<https://polyhaven.com/license>。
