//! 平原高度场预览: 生成灰度高度图 + 统计.
//! 用法: `cargo run -r --bin preview -- [seed] [size_px] [step] [out]`
//! 默认: seed=20260904 size=1024 step=4 out=target/plains.png (覆盖 4096x4096 格).

use std::time::Instant;

use mc::world::plains::PlainsHeightmap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2026_0904);
    let size: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let step: i64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let out = args
        .get(4)
        .cloned()
        .unwrap_or_else(|| "target/plains.png".to_string());

    let gen_map = PlainsHeightmap::new(seed);
    let t0 = Instant::now();
    let n = size as usize;
    let mut h = vec![0f64; n * n];
    let half = size as i64 * step / 2;
    for py in 0..size {
        for px in 0..size {
            let x = px as i64 * step - half;
            let z = py as i64 * step - half;
            h[py as usize * n + px as usize] = gen_map.height(x, z);
        }
    }
    let sample_ms = t0.elapsed().as_millis();

    // 统计.
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for &v in &h {
        min = min.min(v);
        max = max.max(v);
        sum += v;
    }
    let mean = sum / h.len() as f64;
    let var = h.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / h.len() as f64;
    let std = var.sqrt();

    // 坡度: 中心差分 (格/格).
    let at = |x: usize, y: usize| h[y * n + x];
    let mut slope_sum = 0.0;
    let mut steep = 0u64; // 坡度 > 0.35 (~19°) 的像素数: 平原不应多.
    let mut cnt = 0u64;
    for y in 1..n - 1 {
        for x in 1..n - 1 {
            let dx = (at(x + 1, y) - at(x - 1, y)) / (2.0 * step as f64);
            let dz = (at(x, y + 1) - at(x, y - 1)) / (2.0 * step as f64);
            let s = (dx * dx + dz * dz).sqrt();
            slope_sum += s;
            cnt += 1;
            if s > 0.35 {
                steep += 1;
            }
        }
    }
    let mean_slope = slope_sum / cnt as f64;

    // 灰度图: min->黑, max->白.
    let span = (max - min).max(1e-9);
    let img = image::GrayImage::from_fn(size, size, |x, y| {
        let v = h[y as usize * n + x as usize];
        image::Luma([((v - min) / span * 255.0) as u8])
    });
    img.save(&out).expect("保存 PNG 失败");

    println!(
        "seed={seed} size={size}px step={step} 覆盖={}x{}格 -> {out}",
        size as i64 * step,
        size as i64 * step
    );
    println!(
        "采样耗时: {sample_ms}ms ({:.1}μs/列)",
        sample_ms as f64 * 1000.0 / h.len() as f64
    );
    println!(
        "min={min:.2} max={max:.2} 极差={:.2} mean={mean:.2} std={std:.2}",
        max - min
    );
    println!(
        "平均坡度={mean_slope:.3}格/格 陡坡(>0.35)占比={steep_pct:.2}%",
        steep_pct = steep as f64 * 100.0 / cnt as f64
    );
    let steep_frac = steep as f64 * 100.0 / cnt as f64;
    println!(
        "验收: 极差<8? {}  std<2? {}  陡坡<2%? {}",
        max - min < 8.0,
        std < 2.0,
        steep_frac < 2.0
    );
}
