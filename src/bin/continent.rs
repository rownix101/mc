//! 大陆预览: 大尺度海陆图 + 海岸悬崖 zoom.
//! 用法: `cargo run -r --bin continent -- [seed]`
//! 输出: target/continent.png (65536² 格海陆) + target/coast.png (海岸 zoom).

use std::time::Instant;

use mc::world::continent::{SEA_LEVEL, WorldHeightmap};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2026_0904);

    let world = WorldHeightmap::new(seed);

    // 1. 大尺度海陆图: 1024², step=64 → 65536 格.
    let (size, step) = (1024u32, 64i64);
    let n = size as usize;
    let t0 = Instant::now();
    let mut h = vec![0f64; n * n];
    let half = size as i64 * step / 2;
    for py in 0..size {
        for px in 0..size {
            h[py as usize * n + px as usize] =
                world.height(px as i64 * step - half, py as i64 * step - half);
        }
    }
    println!("大尺度采样: {}ms", t0.elapsed().as_millis());

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let (mut land, mut sea) = (0u64, 0u64);
    for &v in &h {
        min = min.min(v);
        max = max.max(v);
        if v > SEA_LEVEL {
            land += 1;
        } else {
            sea += 1;
        }
    }
    println!(
        "min={min:.1} max={max:.1} 陆地占比={:.1}%",
        land as f64 * 100.0 / (land + sea) as f64
    );

    let img = image::RgbImage::from_fn(size, size, |x, y| {
        let v = h[y as usize * n + x as usize];
        if v > SEA_LEVEL {
            // 陆地: 低绿 → 高棕白.
            let t = ((v - SEA_LEVEL) / 12.0).clamp(0.0, 1.0);
            image::Rgb([
                (60.0 + 150.0 * t) as u8,
                (140.0 - 20.0 * t) as u8,
                (60.0 + 60.0 * t) as u8,
            ])
        } else {
            // 海洋: 浅蓝 → 深蓝.
            let t = ((SEA_LEVEL - v) / 16.0).clamp(0.0, 1.0);
            image::Rgb([
                (60.0 - 50.0 * t) as u8,
                (140.0 - 100.0 * t) as u8,
                (200.0 - 80.0 * t) as u8,
            ])
        }
    });
    img.save("target/continent.png").expect("保存失败");
    println!("-> target/continent.png (65536x65536 格)");

    // 2. 找真大洋海岸: field 跌破 c_lo 且此后 4096 格一直是海 (排除内陆湖).
    //    内陆湖 field 为陆, 高度却低于海平面 —— 按高度找会被湖骗.
    let cc = world.continent.config();
    let f = |x: i64| world.continent.field(x as f64, 0.0);
    let mut cx = None;
    let mut x = -half;
    while x <= half {
        if f(x) < cc.c_lo {
            let mut stays_sea = true;
            let mut k = 1;
            while k <= 64 {
                if f(x + k * step) > cc.c_hi {
                    stays_sea = false;
                    break;
                }
                k += 1;
            }
            if stays_sea {
                // 回退到过渡带中点 = 岸线中心.
                let mid = (cc.c_lo + cc.c_hi) / 2.0;
                let mut xr = x;
                while xr > -half && f(xr) < mid {
                    xr -= 16;
                }
                cx = Some(xr);
                break;
            }
        }
        x += step;
    }
    let cx = cx.unwrap_or(0);
    println!("大洋海岸 x≈{cx} (field 中点)");

    // 3. 海岸 zoom: 512² step=4 → 2048 格, 灰度 + 海平面等高线红色.
    let (zs, zstep) = (512u32, 4i64);
    let zn = zs as usize;
    let mut zh = vec![0f64; zn * zn];
    let mut zmin = f64::INFINITY;
    let mut zmax = f64::NEG_INFINITY;
    let mut worst = 0.0f64;
    for py in 0..zs {
        for px in 0..zs {
            let wx = cx + px as i64 * zstep - zs as i64 * zstep / 2;
            let wz = py as i64 * zstep - zs as i64 * zstep / 2;
            let v = world.height(wx, wz);
            zh[py as usize * zn + px as usize] = v;
            zmin = zmin.min(v);
            zmax = zmax.max(v);
        }
    }
    // zoom 区相邻坡度.
    for y in 0..zn - 1 {
        for xx in 0..zn - 1 {
            let d = (zh[y * zn + xx + 1] - zh[y * zn + xx]).abs() / zstep as f64;
            worst = worst.max(d);
        }
    }
    println!(
        "coast zoom: min={zmin:.1} max={zmax:.1} 极差={:.1} 最陡单格坡度={worst:.2}格/格",
        zmax - zmin
    );
    let span = (zmax - zmin).max(1e-9);
    let zimg = image::RgbImage::from_fn(zs, zs, |px, py| {
        let v = zh[py as usize * zn + px as usize];
        // 穿过海平面的像素染红 (岸线).
        let mut cross = false;
        if px + 1 < zs {
            let w = zh[py as usize * zn + (px + 1) as usize];
            if (v > SEA_LEVEL) != (w > SEA_LEVEL) {
                cross = true;
            }
        }
        if cross {
            return image::Rgb([255, 40, 40]);
        }
        if v > SEA_LEVEL {
            let t = ((v - SEA_LEVEL) / (zmax - SEA_LEVEL).max(1.0)) * 255.0;
            image::Rgb([t as u8, t as u8, t as u8])
        } else {
            let t = ((v - zmin) / span * 160.0 + 40.0) as u8;
            image::Rgb([20, 60, t])
        }
    });
    zimg.save("target/coast.png").expect("保存失败");
    println!("-> target/coast.png (2048x2048 格, 红=岸线)");
}
