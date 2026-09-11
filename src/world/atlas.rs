//! 纹理 atlas: 把 `assets/textures/` 下注册的 512x512 PBR 贴图拼成大图.
//!
//! - Tile 顺序 = `Block::unique_textures()` 顺序, 稳定不变 (mesh 里存 TileId).
//! - 网格-packed: `cols = ceil(sqrt(n))`, 行优先.
//! - 输入 PNG 必须是 512x512, 会统一转换为 RGBA.
//! - 每格带 8px 扩边，配合线性过滤和 mipmap 避免材质串色。
//! - `_normal.png` / `_roughness.png` 伴随图与颜色图使用相同布局。
//! - 草方块侧面资源是透明的草皮覆盖层, 构建时叠在泥土底图上.

use image::GenericImage;
use std::path::Path;

use super::block::Block;

pub const TILE: u32 = 512;
pub const GUTTER: u32 = 8;
pub const STRIDE: u32 = TILE + GUTTER * 2;

#[derive(Clone)]
pub struct Atlas {
    /// 每边像素数 (正方形).
    pub size: u32,
    pub cols: u32,
    pub image: image::RgbaImage,
    pub normal_image: image::RgbaImage,
    pub roughness_image: image::RgbaImage,
    pub tiles: Vec<&'static str>,
}

impl Atlas {
    pub fn tile_index(&self, name: &str) -> Option<usize> {
        self.tiles.iter().position(|t| *t == name)
    }

    /// Tile 的 UV 矩形 `(u0, v0, u1, v1)`, `v0` 为上边缘.
    /// UV 延伸到 tile 边界；周围复制的 gutter 为线性过滤提供安全采样区。
    pub fn uv(&self, index: usize) -> (f32, f32, f32, f32) {
        self.uv_region(index, 0, 0, 1, 1)
    }

    /// 取 tile 内的一个规则子区域。自然材质让相邻方块依次使用 2x2
    /// 子区域，使每块保有 256px 细节，同时连续铺开而不是逐格重复。
    pub fn uv_region(
        &self,
        index: usize,
        region_x: u32,
        region_y: u32,
        columns: u32,
        rows: u32,
    ) -> (f32, f32, f32, f32) {
        let s = self.size as f32;
        let tile_x = (index as u32 % self.cols) * STRIDE + GUTTER;
        let tile_y = (index as u32 / self.cols) * STRIDE + GUTTER;
        let region_width = TILE / columns;
        let region_height = TILE / rows;
        let x = tile_x + (region_x % columns) * region_width;
        let y = tile_y + (region_y % rows) * region_height;
        (
            x as f32 / s,
            y as f32 / s,
            (x + region_width) as f32 / s,
            (y + region_height) as f32 / s,
        )
    }

    /// 方块某面的 TileId (= 在 `tiles` 中的下标).
    pub fn tile_id(&self, block: Block, tile: &str) -> usize {
        let _ = block;
        self.tile_index(tile).expect("贴图不在 atlas 中")
    }
}

/// 从目录加载全部去重贴图并拼 atlas.
pub fn build(dir: &Path) -> Result<Atlas, String> {
    let tiles = Block::unique_textures();
    let n = tiles.len() as u32;
    let cols = n.isqrt() + u32::from(n.isqrt() * n.isqrt() != n);
    let rows = n.div_ceil(cols);
    let size = cols.max(rows) * STRIDE;
    let mut image = image::RgbaImage::new(size, size);
    let mut normal_image = image::RgbaImage::new(size, size);
    let mut roughness_image = image::RgbaImage::new(size, size);
    for (i, name) in tiles.iter().enumerate() {
        // `grass_block_side_overlay.png` 只有顶部草皮, 透明区域并不是方块的
        // 空洞。先铺泥土再叠加草皮, 避免透明像素露出天空或物品栏背景。
        let load_composited = |kind: MapKind| -> Result<image::RgbaImage, String> {
            if *name != Block::GrassBlock.def().side {
                return load_tile(dir, name, kind);
            }
            let mut dirt = load_tile(dir, Block::Dirt.def().side, kind)?;
            let grass = load_tile(dir, name, kind)?;
            image::imageops::overlay(&mut dirt, &grass, 0, 0);
            // 这是一个实体方块, 即使来源贴图带有半透明边缘也不能
            // 把背景带入 atlas。
            for pixel in dirt.pixels_mut() {
                pixel[3] = u8::MAX;
            }
            Ok(dirt)
        };
        let tile = load_composited(MapKind::Color)?;
        let normal = load_composited(MapKind::Normal)?;
        let roughness = load_composited(MapKind::Roughness)?;

        let x = (i as u32 % cols) * STRIDE;
        let y = (i as u32 / cols) * STRIDE;
        copy_with_gutter(&mut image, &tile, x, y).map_err(|e| format!("拼 {name} 失败: {e}"))?;
        copy_with_gutter(&mut normal_image, &normal, x, y)
            .map_err(|e| format!("拼 {name} 法线失败: {e}"))?;
        copy_with_gutter(&mut roughness_image, &roughness, x, y)
            .map_err(|e| format!("拼 {name} 粗糙度失败: {e}"))?;
    }
    Ok(Atlas {
        size,
        cols,
        image,
        normal_image,
        roughness_image,
        tiles,
    })
}

#[derive(Clone, Copy)]
enum MapKind {
    Color,
    Normal,
    Roughness,
}

fn map_name(name: &str, kind: MapKind) -> String {
    let suffix = match kind {
        MapKind::Color => return name.to_owned(),
        MapKind::Normal => "_normal.png",
        MapKind::Roughness => "_roughness.png",
    };
    format!("{}{suffix}", name.strip_suffix(".png").unwrap_or(name))
}

fn load_tile(dir: &Path, name: &str, kind: MapKind) -> Result<image::RgbaImage, String> {
    let mapped_name = map_name(name, kind);
    let tile = image::open(dir.join(&mapped_name))
        .map_err(|e| format!("打开 {mapped_name} 失败: {e}"))?
        .to_rgba8();
    if tile.width() != TILE || tile.height() != TILE {
        return Err(format!(
            "{mapped_name} 尺寸 {:?}, 必须 {TILE}x{TILE}",
            (tile.width(), tile.height())
        ));
    }
    Ok(tile)
}

fn copy_with_gutter(
    atlas: &mut image::RgbaImage,
    tile: &image::RgbaImage,
    x: u32,
    y: u32,
) -> image::ImageResult<()> {
    atlas.copy_from(tile, x + GUTTER, y + GUTTER)?;
    for offset in 0..GUTTER {
        for tx in 0..TILE {
            atlas.put_pixel(x + GUTTER + tx, y + offset, *tile.get_pixel(tx, 0));
            atlas.put_pixel(
                x + GUTTER + tx,
                y + GUTTER + TILE + offset,
                *tile.get_pixel(tx, TILE - 1),
            );
        }
        for ty in 0..TILE {
            atlas.put_pixel(x + offset, y + GUTTER + ty, *tile.get_pixel(0, ty));
            atlas.put_pixel(
                x + GUTTER + TILE + offset,
                y + GUTTER + ty,
                *tile.get_pixel(TILE - 1, ty),
            );
        }
    }
    for gy in 0..GUTTER {
        for gx in 0..GUTTER {
            atlas.put_pixel(x + gx, y + gy, *tile.get_pixel(0, 0));
            atlas.put_pixel(x + GUTTER + TILE + gx, y + gy, *tile.get_pixel(TILE - 1, 0));
            atlas.put_pixel(x + gx, y + GUTTER + TILE + gy, *tile.get_pixel(0, TILE - 1));
            atlas.put_pixel(
                x + GUTTER + TILE + gx,
                y + GUTTER + TILE + gy,
                *tile.get_pixel(TILE - 1, TILE - 1),
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    #[test]
    fn atlas_builds_and_indexed() {
        let a = build(Path::new("assets/textures")).expect("atlas 构建失败");
        // 33 tiles → cols=6，tile 512px + 两侧各 8px gutter。
        assert_eq!(a.tiles.len(), 33);
        assert_eq!(a.cols, 6);
        assert_eq!(a.size, 3168);
        assert_eq!(a.tile_index("grass_block_top.png"), Some(0));
        assert_eq!(a.tile_index("dirt.png"), Some(2));
        assert_eq!(a.tile_index("water_overlay.png"), Some(22));
        assert_eq!(a.tile_index("piston_inner.png"), Some(31));
        let grass_side = a.tile_index(Block::GrassBlock.def().side).unwrap();
        let x = (grass_side as u32 % a.cols) * STRIDE + GUTTER;
        let y = (grass_side as u32 / a.cols) * STRIDE + GUTTER;
        assert!(
            a.image
                .view(x, y, TILE, TILE)
                .pixels()
                .all(|(_, _, p)| p[3] == 255)
        );
        assert_eq!(a.tile_index("nope.png"), None);
        // UV 落在对应格内.
        let (u0, v0, u1, v1) = a.uv(0);
        assert!(u0 < u1 && v0 < v1);
        assert!(u1 <= (GUTTER + TILE) as f32 / a.size as f32 + 1e-6);
        let full = a.uv(0);
        let half = a.uv_region(0, 0, 0, 2, 2);
        assert!(half.2 - half.0 < full.2 - full.0);
        // 原木顶/侧是不同 tile.
        let top = a.tile_id(Block::OakLog, Block::OakLog.def().top);
        let side = a.tile_id(Block::OakLog, Block::OakLog.def().side);
        assert_ne!(top, side);
    }
}
