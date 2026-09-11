//! 地表铺贴对照预览: 看自然材质的贴图在相邻方块上是怎么接起来的.
//!
//! 每列一种材质，上下分别检查传统坐标和当前网格 UV 的连续性。
//! 此 CPU 预览不包含片元着色器里的沙地随机混合；最终外观以 GPU 渲染为准。
//!
//! 用法: `cargo run --release --bin tiling [每格像素]`
//! 输出: `target/tiling.png`

use std::path::Path;

use image::imageops::{self, FilterType};
use image::{Rgba, RgbaImage};
use mc::render::voxel::face_texture;
use mc::world::atlas::{Atlas, build};
use mc::world::block::{Block, Face};

const PATCH: u32 = 8;
const MATERIALS: [Block; 6] = [
    Block::GrassBlock,
    Block::Dirt,
    Block::Stone,
    Block::Cobblestone,
    Block::Sand,
    Block::Gravel,
];
const GAP: u32 = 8;

const TOP: [f32; 3] = [0.0, 1.0, 0.0];

/// 旧规则 (只有顶面, 没有镜像): U 轴 = x, V 轴 = -z, 取样区域看奇偶.
fn legacy_region(x: i64, z: i64) -> ([u32; 4], bool, bool) {
    (
        [x.rem_euclid(2) as u32, (-z).rem_euclid(2) as u32, 2, 2],
        false,
        false,
    )
}

/// 把某个方块面的贴图裁出来, 按需镜像, 缩放到 `size` 见方.
fn tile_pixels(
    atlas: &Atlas,
    block: Block,
    x: i64,
    y: i64,
    z: i64,
    size: u32,
    legacy: bool,
) -> RgbaImage {
    let tile = atlas.tile_id(block, block.tile(Face::Top));
    let (region, mirror_u, mirror_v) = if legacy {
        legacy_region(x, z)
    } else {
        let texture = face_texture(block, Face::Top, TOP, x, y, z);
        (texture.region, texture.mirror_u, texture.mirror_v)
    };
    let scale = atlas.size as f32;
    let (u0, v0, u1, v1) = atlas.uv_region(tile, region[0], region[1], region[2], region[3]);
    let (px0, py0) = ((u0 * scale).round() as u32, (v0 * scale).round() as u32);
    let (px1, py1) = ((u1 * scale).round() as u32, (v1 * scale).round() as u32);
    let mut patch = imageops::crop_imm(&atlas.image, px0, py0, px1 - px0, py1 - py0).to_image();
    if mirror_u {
        imageops::flip_horizontal_in_place(&mut patch);
    }
    if mirror_v {
        imageops::flip_vertical_in_place(&mut patch);
    }
    imageops::resize(&patch, size, size, FilterType::Triangle)
}

/// 画一块 `PATCH`×`PATCH` 的地表; `legacy` 为真时用旧规则.
fn patch_image(atlas: &Atlas, block: Block, size: u32, legacy: bool) -> RgbaImage {
    let mut image = RgbaImage::new(PATCH * size, PATCH * size);
    for bz in 0..PATCH as i64 {
        for bx in 0..PATCH as i64 {
            let tile = tile_pixels(atlas, block, bx, 0, bz, size, legacy);
            imageops::overlay(&mut image, &tile, bx * size as i64, bz * size as i64);
        }
    }
    image
}

/// 相隔 2 格的单元之间平均差多少. 旧规则严格 2 格周期, 这个值接近 0.
fn two_block_repetition(patch: &RgbaImage, size: u32) -> f64 {
    let cell = size as i64;
    let mut sum = 0.0;
    let mut count = 0.0;
    for y in 0..(PATCH as i64 - 2) * cell {
        for x in 0..(PATCH as i64 - 2) * cell {
            let a = patch.get_pixel(x as u32, y as u32);
            let b = patch.get_pixel((x + 2 * cell) as u32, (y + 2 * cell) as u32);
            sum += (0..3)
                .map(|c| (a[c] as f64 - b[c] as f64).abs())
                .sum::<f64>();
            count += 3.0;
        }
    }
    sum / count
}

fn main() {
    let size: u32 = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(96)
        .clamp(16, 512);
    let atlas = build(Path::new("assets/textures")).expect("atlas 构建失败");

    let column = PATCH * size;
    let width = MATERIALS.len() as u32 * column + (MATERIALS.len() as u32 - 1) * GAP;
    let height = column * 2 + GAP;
    let mut canvas = RgbaImage::from_pixel(width, height, Rgba([20, 24, 36, 255]));

    for (index, block) in MATERIALS.iter().enumerate() {
        let x0 = index as u32 * (column + GAP);
        for (row, legacy) in [true, false].into_iter().enumerate() {
            let patch = patch_image(&atlas, *block, size, legacy);
            let y0 = row as u32 * (column + GAP);
            imageops::overlay(&mut canvas, &patch, x0 as i64, y0 as i64);
            println!(
                "{:16} {:5}: 2 格自相似度(0=完全重复) = {:.2}",
                block.def().name,
                if legacy { "旧" } else { "新" },
                two_block_repetition(&patch, size)
            );
        }
    }

    canvas.save("target/tiling.png").expect("保存失败");
    println!(
        "-> target/tiling.png  ({}×{}, 每格 {size}px)\n   上排=旧规则(严格 2 格重复), 下排=当前网格 UV（不含 GPU 沙地混合）",
        width, height
    );
}
