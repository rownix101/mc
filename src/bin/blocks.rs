//! 方块预览: atlas 拼图 + 剖面图.
//! 用法: `cargo run -r --bin blocks`
//! 输出: `target/atlas.png` (5x5 tile 网格) + `target/column.png` (陆/滩/海三柱剖面).

use std::path::Path;

use mc::world::atlas::{TILE, build};
use mc::world::block::{Block, Face};
use mc::world::column::{ColumnGen, SEA_Y};
use mc::world::continent::{SEA_LEVEL, WorldHeightmap};
use mc::world::tree::TreeDecorator;

/// 方块 → 展示用贴图 (表面优先, 原木用侧面).
fn show_tile(b: Block) -> &'static str {
    match b {
        Block::OakLog => b.def().side,
        Block::Water | Block::Glass | Block::OakLeaves => b.def().side,
        _ => b.def().top,
    }
}

fn main() {
    // 1. atlas.
    let atlas = build(Path::new("assets/textures")).expect("atlas 构建失败");
    atlas
        .image
        .save("target/atlas.png")
        .expect("保存 atlas 失败");
    println!(
        "atlas: {} tiles, {}x{}px -> target/atlas.png",
        atlas.tiles.len(),
        atlas.size,
        atlas.size
    );
    for b in mc::world::block::ALL {
        if b == Block::Air {
            continue;
        }
        let top = atlas.tile_id(b, b.tile(Face::Top));
        let side = atlas.tile_id(b, b.tile(Face::Side));
        let bot = atlas.tile_id(b, b.tile(Face::Bottom));
        println!(
            "  id={:2} {:16} top={top:2} side={side:2} bottom={bot:2} {:?}",
            b.id(),
            b.def().name,
            b.def().render
        );
    }

    // 2. 三柱剖面: 真实高度场找陆/滩/海各一点.
    let world = WorldHeightmap::new(2026_0904);
    let colgen = ColumnGen::new(2026_0904);
    let treegen = TreeDecorator::new(2026_0904);
    type Sample = Option<(i64, i64)>;
    let (mut land, mut beach, mut sea): (Sample, Sample, Sample) = (None, None, None);
    let mut z = 0i64;
    while (land.is_none() || beach.is_none() || sea.is_none()) && z < 200_000 {
        for x in (-20000..20000).step_by(500) {
            let h = world.height(x, z);
            if h > SEA_LEVEL + 2.0 && land.is_none() {
                land = Some((x, z));
            } else if h > SEA_LEVEL && h <= SEA_LEVEL + 2.0 && beach.is_none() {
                beach = Some((x, z));
            } else if h < SEA_LEVEL - 4.0 && sea.is_none() {
                sea = Some((x, z));
            }
        }
        z += 997;
    }
    let cols = [land.unwrap(), beach.unwrap(), sea.unwrap()];
    let tex = |name: &str| {
        image::open(Path::new("assets/textures").join(name))
            .expect("贴图缺失")
            .to_rgba8()
    };
    let tiles: std::collections::HashMap<&str, image::RgbaImage> =
        atlas.tiles.iter().map(|t| (*t, tex(t))).collect();

    // 每柱宽 64px (4 tiles), 高 128 格.
    let cw = TILE * 4;
    let ch = 128 * 8;
    let mut img = image::RgbaImage::new(cw * 3 + 32, ch);
    for (ci, (x, z)) in cols.iter().enumerate() {
        let h = world.height(*x, *z);
        let col = colgen.generate(*x, *z, h);
        let top = h.floor() as i64;
        // 在该柱周围生成树, 取覆盖该柱的树块.
        let tree_region = treegen.generate(&world, *x - 4, *x + 5, *z - 4, *z + 5);
        let mut tree_blocks: std::collections::HashMap<(i64, i64, i64), Block> =
            std::collections::HashMap::new();
        for tree in &tree_region {
            for (bx, by, bz, b) in tree.blocks() {
                if bx == *x && bz == *z {
                    tree_blocks.insert((bx, by, bz), b);
                }
            }
        }
        println!(
            "柱{ci}: ({x},{z}) h={h:.1} top={top} 表面={:?} 树块={}",
            col.get(top),
            tree_blocks.len()
        );
        // y=0 在底, 每格 8px.
        for y in 0..=127 {
            // 优先用树块, 否则用柱块.
            let b = tree_blocks
                .get(&(*x, y as i64, *z))
                .copied()
                .unwrap_or_else(|| col.get(y as i64));
            let tile_img = if b == Block::Air {
                None
            } else {
                Some(&tiles[show_tile(b)])
            };
            let oy = ch - (y as u32 + 1) * 8;
            for px in 0..cw {
                for py in 0..8u32 {
                    let p = if let Some(t) = tile_img {
                        let tx = px * 16 / cw;
                        let ty = py * 16 / 8;
                        *t.get_pixel(tx, ty)
                    } else {
                        image::Rgba([20, 24, 36, 255])
                    };
                    img.put_pixel(ci as u32 * (cw + 16) + px, oy + py, p);
                }
            }
        }
        // 海平面线.
        let oy = ch - (SEA_Y as u32 + 1) * 8;
        for px in 0..cw {
            img.put_pixel(
                ci as u32 * (cw + 16) + px,
                oy,
                image::Rgba([255, 60, 60, 255]),
            );
        }
    }
    // atlas 拼图也附一张 tile 索引图? 已有 atlas.png, 不再重复.
    img.save("target/column.png").expect("保存失败");
    println!("-> target/column.png (左陆/中滩/右海, 红线=海平面 y=60)");
}
