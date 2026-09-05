//! 平原高度场原型 (Epic Terrain 风格解析式侵蚀路线).
//!
//! 纯函数 `h(x, z)`, 细节见 `PlainsHeightmap::height`.
//!
//! ```text
//! h = base + broad * A_b + micro * A_m - valley_carve * mask
//! ```
//! - `broad`: OpenSimplex2 FBm 3 oct, 波长 ~700m, 振幅 ±2.0 格 (大尺度起伏)
//! - `micro`: ValueCubic FBm 2 oct, 波长 ~50m, 振幅 ±0.35 格 (细节, 不碎面)
//! - `valley`: OpenSimplex2 FBm **2 oct**, 波长 ~950m, `|fbm|` 零线作谷线,
//!   smoothstep 剖面 carve, 深 ~3 格, 半宽 ~38 格 (宽浅, 长期侵蚀感).
//!   octave 从 3 降到 2: 去掉高频小虫纹, 只留大尺度树枝谷.
//! - `mask`: OpenSimplex2 FBm 2 oct, 波长 ~2200m, smoothstep 门限,
//!   谷只在 ~40% 区域出现 (排水盆地感), 其余为完整平坦面.
//! - `warp`: 两路单 octave OpenSimplex2, 波长 ~1100m, ±180 格,
//!   只弯曲大尺度谷线 (波长需 ≥ 谷波长, 否则会把谷线揉碎).
//!
//! 约束: 总起伏 < 8 格; 每列 6 次 `get_noise_2d`.

use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

/// 平原可调参数. 后续山脉/河流/海岸都是加项, 不改基底.
#[derive(Clone, Debug)]
pub struct PlainsConfig {
    /// 基准高度 (格).
    pub base: f64,
    /// 大尺度起伏振幅 (格). FBm 输出 [-1, 1].
    pub broad_amp: f64,
    /// 微细节振幅 (格).
    pub micro_amp: f64,
    /// 谷地最大深度 (格), 实际深度会被 broad 调制到 ~0.8x--1.2x.
    pub valley_depth: f64,
    /// 谷线半宽 (格). 剖面 smoothstep, 谷底平坦段约占半宽.
    pub valley_half_width: f64,
    /// 谷线弯曲强度 (格).
    pub warp_amp: f64,
    /// 谷覆盖率 0--1. mask 场高于门限才 carve, 典型 0.4.
    pub valley_coverage: f64,
    pub seed: u64,
}

impl Default for PlainsConfig {
    fn default() -> Self {
        Self {
            base: 64.0,
            broad_amp: 2.2,
            micro_amp: 0.35,
            valley_depth: 2.5,
            valley_half_width: 230.0,
            warp_amp: 180.0,
            valley_coverage: 0.6,
            seed: 2026_0904,
        }
    }
}

pub struct PlainsHeightmap {
    config: PlainsConfig,
    broad: FastNoiseLite,
    micro: FastNoiseLite,
    valley: FastNoiseLite,
    mask: FastNoiseLite,
    warp_x: FastNoiseLite,
    warp_z: FastNoiseLite,
}

impl PlainsHeightmap {
    pub fn new(seed: u64) -> Self {
        Self::with_config(PlainsConfig {
            seed,
            ..PlainsConfig::default()
        })
    }

    pub fn with_config(config: PlainsConfig) -> Self {
        let mut s = splitmix64(config.seed);
        let mut next_seed = || {
            s = splitmix64(s);
            (s & 0x7fff_ffff) as i32
        };
        // 注意: frequency 是 f32 (上游 fastnoise-lite 字段类型), 即使开了 f64 feature
        // 采样坐标才是 f64. 波长 = 1 / frequency (格).
        let broad = make_noise(
            next_seed(),
            NoiseType::OpenSimplex2,
            FractalType::FBm,
            3,
            1.0 / 700.0,
        );
        let micro = make_noise(
            next_seed(),
            NoiseType::ValueCubic,
            FractalType::FBm,
            2,
            1.0 / 50.0,
        );
        // 谷线场: 2 oct 即可, 高频 octave 是虫纹元凶.
        let valley = make_noise(
            next_seed(),
            NoiseType::OpenSimplex2,
            FractalType::FBm,
            2,
            1.0 / 950.0,
        );
        // mask 场: 超低频, 决定哪里有谷.
        let mask = make_noise(
            next_seed(),
            NoiseType::OpenSimplex2,
            FractalType::FBm,
            2,
            1.0 / 2200.0,
        );
        // warp 用单 octave、无 fractal, 输出 [-1, 1] 直接乘格数做偏移.
        // 波长必须 ≥ 谷波长, 否则谷线被揉碎成虫纹.
        let warp_x = make_noise(
            next_seed(),
            NoiseType::OpenSimplex2,
            FractalType::None,
            1,
            1.0 / 1100.0,
        );
        let warp_z = make_noise(
            next_seed(),
            NoiseType::OpenSimplex2,
            FractalType::None,
            1,
            1.0 / 1100.0,
        );
        Self {
            config,
            broad,
            micro,
            valley,
            mask,
            warp_x,
            warp_z,
        }
    }

    pub fn config(&self) -> &PlainsConfig {
        &self.config
    }

    /// 权威采样: 输入 i64 方块坐标, f64 全程计算 (远坐标稳定).
    pub fn height(&self, x: i64, z: i64) -> f64 {
        self.height_f64(x as f64, z as f64)
    }

    pub(crate) fn height_f64_pub(&self, x: f64, z: f64) -> f64 {
        self.height_f64(x, z)
    }

    fn height_f64(&self, x: f64, z: f64) -> f64 {
        let c = &self.config;
        // 1. warp 谷线 (手动偏移, ±warp_amp 格, 量级可控).
        let wx = x + c.warp_amp * self.warp_x.get_noise_2d(x, z) as f64;
        let wz = z + c.warp_amp * self.warp_z.get_noise_2d(x, z) as f64;
        // 2. 基底起伏.
        let b = self.broad.get_noise_2d(x, z) as f64;
        let m = self.micro.get_noise_2d(x, z) as f64;
        // 3. 谷线: |fbm| 在 0 处为谷心线 (树枝状零等值线).
        //    FBm 输出已归一化到 [-1, 1]; 波长 950m, 零线附近梯度量级 ~1/950,
        //    谷半宽 hw 格 → 噪声域阈值 hw/950.
        let v = self.valley.get_noise_2d(wx, wz) as f64;
        let d = v.abs();
        let hw_world = c.valley_half_width * (1.0 + 0.25 * b);
        let t = (1.0 - d * (950.0 / hw_world)).clamp(0.0, 1.0);
        let profile = t * t * (3.0 - 2.0 * t);
        // 4. mask: 只有部分区域有谷. mask 场 [-1,1] → 覆盖率映射.
        //    coverage=0.4 → 门限≈0.2: mask 高的 40% 区域 carve.
        let mk = self.mask.get_noise_2d(x, z) as f64;
        let threshold = 1.0 - 2.0 * c.valley_coverage;
        let mt = ((mk - threshold) / 0.35).clamp(0.0, 1.0);
        let mask_s = mt * mt * (3.0 - 2.0 * mt);
        // 低地谷更深 (自然排水), 深度 ~2.4--3.6 格.
        let depth = c.valley_depth * (1.0 - 0.2 * b);
        c.base + b * c.broad_amp + m * c.micro_amp - profile * mask_s * depth
    }
}

pub fn make_noise(
    seed: i32,
    noise_type: NoiseType,
    fractal: FractalType,
    octaves: i32,
    frequency: f32,
) -> FastNoiseLite {
    let mut n = FastNoiseLite::with_seed(seed);
    n.set_noise_type(Some(noise_type));
    n.set_fractal_type(Some(fractal));
    n.set_fractal_octaves(Some(octaves));
    n.set_fractal_lacunarity(Some(2.0));
    n.set_fractal_gain(Some(0.5));
    n.set_frequency(Some(frequency));
    n
}

/// smoothstep 边函数: x<=lo→0, x>=hi→1, 中间三次平滑.
pub fn smoothstep(lo: f64, hi: f64, x: f64) -> f64 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// splitmix64, world seed (u64) → 各通道 i32 seed.
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let g = PlainsHeightmap::new(1234);
        assert_eq!(g.height(0, 0), g.height(0, 0));
        assert_eq!(g.height(-999_999, 888_888), g.height(-999_999, 888_888));
    }

    #[test]
    fn different_seeds_differ() {
        let a = PlainsHeightmap::new(1);
        let b = PlainsHeightmap::new(2);
        let mut same = 0;
        for i in 0..16 {
            if a.height(i * 37, i * -53) == b.height(i * 37, i * -53) {
                same += 1;
            }
        }
        assert!(same < 16, "不同 seed 应产生不同地形");
    }

    #[test]
    fn finite_and_sane_range() {
        let g = PlainsHeightmap::new(PlainsConfig::default().seed);
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut n = 0u32;
        let mut i = 0i64;
        while i < 2048 {
            let h = g.height(i, -i / 2);
            assert!(h.is_finite(), "高度必须有限: ({i})");
            min = min.min(h);
            max = max.max(h);
            n += 1;
            i += 8;
        }
        assert!(n > 200);
        // 真平原: 均值附近小范围浮动, 极值也不应离谱.
        assert!(min > 52.0 && max < 74.0, "超出平原合理范围: [{min}, {max}]");
        assert!(max - min < 12.0, "极值跨度过大: {}", max - min);
    }

    #[test]
    fn carve_never_raises_terrain() {
        // 谷地项只能下切: h <= base + broad + micro (即 carve >= 0).
        let g = PlainsHeightmap::new(7);
        for i in 0..512 {
            let (x, z) = (i * 13 - 3000, i * 29 + 1000);
            let h = g.height(x, z);
            assert!(h <= 64.0 + 2.6 + 0.45 + 1e-6, "carve 不应抬高地形: {h}");
        }
    }
}
