//! 纹理 atlas: 把 `assets/textures/` 下 21 张 16x16 REFI 贴图拼成一张大图.
//!
//! - Tile 顺序 = `Block::unique_textures()` 顺序, 稳定不变 (mesh 里存 TileId).
//! - 网格-packed: `cols = ceil(sqrt(n))`, 行优先.
//! - 输入 PNG 必须是 16x16 RGBA (已验证全是).
//! - 草方块侧面资源是透明的草皮覆盖层, 构建时叠在泥土底图上.

use image::GenericImage;
use std::path::Path;

use super::block::Block;

pub const TILE: u32 = 16;

#[derive(Clone)]
pub struct Atlas {
    /// 每边像素数 (正方形).
    pub size: u32,
    pub cols: u32,
    pub image: image::RgbaImage,
    pub tiles: Vec<&'static str>,
}

impl Atlas {
    pub fn tile_index(&self, name: &str) -> Option<usize> {
        self.tiles.iter().position(|t| *t == name)
    }

    /// Tile 的 UV 矩形 `(u0, v0, u1, v1)`, `v0` 为上边缘.
    /// 半像素内缩, 避免线性过滤时 bleeding (当前 Nearest 下无所谓, 先留好).
    pub fn uv(&self, index: usize) -> (f32, f32, f32, f32) {
        let s = self.size as f32;
        let x = (index as u32 % self.cols) * TILE;
        let y = (index as u32 / self.cols) * TILE;
        let inset = 0.5;
        (
            (x as f32 + inset) / s,
            (y as f32 + inset) / s,
            (x as f32 + TILE as f32 - inset) / s,
            (y as f32 + TILE as f32 - inset) / s,
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
    let size = cols.max(rows) * TILE;
    let mut image = image::RgbaImage::new(size, size);
    for (i, name) in tiles.iter().enumerate() {
        // `default_grass_side.png` 只有顶部草皮, 透明区域并不是方块的
        // 空洞。先铺泥土再叠加草皮, 避免透明像素露出天空或物品栏背景。
        let tile = if *name == Block::GrassBlock.def().side {
            let mut dirt = load_tile(dir, Block::Dirt.def().side)?;
            let grass = load_tile(dir, name)?;
            image::imageops::overlay(&mut dirt, &grass, 0, 0);
            // 这是一个实体方块, 即使来源贴图带有半透明边缘也不能
            // 把背景带入 atlas。
            for pixel in dirt.pixels_mut() {
                pixel[3] = u8::MAX;
            }
            dirt
        } else {
            load_tile(dir, name)?
        };

        let x = (i as u32 % cols) * TILE;
        let y = (i as u32 / cols) * TILE;
        image
            .copy_from(&tile, x, y)
            .map_err(|e| format!("拼 {name} 失败: {e}"))?;
    }
    Ok(Atlas {
        size,
        cols,
        image,
        tiles,
    })
}

fn load_tile(dir: &Path, name: &str) -> Result<image::RgbaImage, String> {
    let tile = image::open(dir.join(name))
        .map_err(|e| format!("打开 {name} 失败: {e}"))?
        .to_rgba8();
    if tile.width() != TILE || tile.height() != TILE {
        return Err(format!(
            "{name} 尺寸 {:?}, 必须 16x16",
            (tile.width(), tile.height())
        ));
    }
    Ok(tile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    #[test]
    fn atlas_builds_and_indexed() {
        let a = build(Path::new("assets/textures")).expect("atlas 构建失败");
        // 21 tiles → cols=5, size=80.
        assert_eq!(a.tiles.len(), 21);
        assert_eq!(a.cols, 5);
        assert_eq!(a.size, 80);
        assert_eq!(a.tile_index("default_grass.png"), Some(0));
        assert_eq!(a.tile_index("default_dirt.png"), Some(2));
        assert_eq!(a.tile_index("default_water.png"), Some(20));
        let grass_side = a.tile_index(Block::GrassBlock.def().side).unwrap();
        let x = (grass_side as u32 % a.cols) * TILE;
        let y = (grass_side as u32 / a.cols) * TILE;
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
        assert!(u1 <= 16.0 / 80.0 + 1e-6);
        // 原木顶/侧是不同 tile.
        let top = a.tile_id(Block::OakLog, Block::OakLog.def().top);
        let side = a.tile_id(Block::OakLog, Block::OakLog.def().side);
        assert_ne!(top, side);
    }
}
