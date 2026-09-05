//! 大陆板块 mask: 超低频海陆 + 海岸悬崖 carve.
//!
//! 纯函数, 与 `plains` 正交叠加:
//!
//! ```text
//! c     = continent_fbm(warp(x,z)) + shore_detail * 0.05   // 海陆场 [-1,1]
//! land  = smoothstep(C_LO, C_HI, c)                        // 0 海 → 1 陆
//! floor = SEA - 5 - 11*deep + seabed * (1 - land)          // 海底
//! h     = land_h - (land_h - floor) * (1 - land)^P         // P<1: 近岸陡峭
//! ```
//!
//! - `continent`: OpenSimplex2 FBm 2 oct, 波长 ~48000 格 → 万格级海洋/大陆.
//! - `warp`: 两路单 octave, 波长 ~60000 格, ±7500 格 → 大陆形状不圆.
//! - `shore`: 单 octave, 波长 ~900 格, 振幅 0.05(c 单位) → 海湾/岬角 (~250 格位移).
//! - `seabed`: 单 octave ValueCubic, 波长 ~450 格, ±1.5 格 → 陆架质感 (只在海里).
//! - 悬崖: 窄过渡带 (C_HI-C_LO=0.08) + 幂曲线 `(1-land)^P, P=0.18`,
//!   90% 落差集中在岸线陆侧 ~200 格内 → 近岸陡壁 + 外侧陆架缓坡.
//!   单调下降, 无断层、无海沟回弹.
//! - 海平面 `SEA=60.5`: 平原均值 ~63.9, 低谷 (~59) 成湖, 后续接河流.
//!
//! 每列新增 5 次 `get_noise_2d` (warp 2 + continent 1 + shore 1 + seabed 1).

use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

use super::plains::{PlainsConfig, PlainsHeightmap, make_noise, smoothstep};

/// 海平面 (格). 低于此为海/湖.
pub const SEA_LEVEL: f64 = 60.5;

#[derive(Clone, Debug)]
pub struct ContinentConfig {
    /// 海陆过渡带 (c 单位): c<C_LO 全海, c>C_HI 全陆. 非对称: 海侧宽 (陆架), 陆侧窄.
    pub c_lo: f64,
    pub c_hi: f64,
    /// 深海过渡带: c<-0.55 深海底, c>-0.12 陆架.
    pub deep_lo: f64,
    pub deep_hi: f64,
    /// 陆架深度 / 深海追加深度 (格).
    pub shelf_depth: f64,
    pub abyss_depth: f64,
    /// 悬崖幂指数 P<1. 越小越陡. 0.35 ≈ 近岸 56°+ 陡壁 + 外侧陆架缓坡.
    pub cliff_pow: f64,
    /// 岸线细节振幅 (c 单位).
    pub shore_amp: f64,
    /// 海底起伏振幅 (格).
    pub seabed_amp: f64,
    /// 大陆 warp 强度 (格).
    pub warp_amp: f64,
    pub seed: u64,
}

impl Default for ContinentConfig {
    fn default() -> Self {
        Self {
            c_lo: -0.02,
            c_hi: 0.06,
            deep_lo: -0.30,
            deep_hi: -0.06,
            shelf_depth: 5.0,
            abyss_depth: 11.0,
            cliff_pow: 0.18,
            shore_amp: 0.025,
            seabed_amp: 1.5,
            warp_amp: 7500.0,
            seed: 2026_0904,
        }
    }
}

pub struct ContinentMask {
    config: ContinentConfig,
    field: FastNoiseLite,
    shore: FastNoiseLite,
    seabed: FastNoiseLite,
    warp_x: FastNoiseLite,
    warp_z: FastNoiseLite,
}

impl ContinentMask {
    pub fn new(seed: u64) -> Self {
        Self::with_config(ContinentConfig {
            seed,
            ..ContinentConfig::default()
        })
    }

    pub fn with_config(config: ContinentConfig) -> Self {
        let mut s = super::plains::splitmix64(config.seed ^ 0xC0A5_71A7);
        let mut next_seed = || {
            s = super::plains::splitmix64(s);
            (s & 0x7fff_ffff) as i32
        };
        let field = make_noise(
            next_seed(),
            NoiseType::OpenSimplex2,
            FractalType::FBm,
            2,
            1.0 / 48000.0,
        );
        let shore = make_noise(
            next_seed(),
            NoiseType::OpenSimplex2,
            FractalType::None,
            1,
            1.0 / 900.0,
        );
        let seabed = make_noise(
            next_seed(),
            NoiseType::ValueCubic,
            FractalType::None,
            1,
            1.0 / 450.0,
        );
        let warp_x = make_noise(
            next_seed(),
            NoiseType::OpenSimplex2,
            FractalType::None,
            1,
            1.0 / 60000.0,
        );
        let warp_z = make_noise(
            next_seed(),
            NoiseType::OpenSimplex2,
            FractalType::None,
            1,
            1.0 / 60000.0,
        );
        Self {
            config,
            field,
            shore,
            seabed,
            warp_x,
            warp_z,
        }
    }

    pub fn config(&self) -> &ContinentConfig {
        &self.config
    }

    /// 海陆场 c ∈ [-1, 1]. >0 倾向陆, <0 倾向海. 纯函数, f64 全程.
    pub fn field(&self, x: f64, z: f64) -> f64 {
        let c = &self.config;
        let wx = x + c.warp_amp * self.warp_x.get_noise_2d(x, z) as f64;
        let wz = z + c.warp_amp * self.warp_z.get_noise_2d(x, z) as f64;
        self.field.get_noise_2d(wx, wz) as f64 + c.shore_amp * self.shore.get_noise_2d(x, z) as f64
    }

    /// 陆地权重 0--1.
    pub fn land_factor(&self, x: f64, z: f64) -> f64 {
        let c = &self.config;
        smoothstep(c.c_lo, c.c_hi, self.field(x, z))
    }
}

/// 世界高度场 = 大陆 mask × 平原基底. 权威采样入口.
pub struct WorldHeightmap {
    pub plains: PlainsHeightmap,
    pub continent: ContinentMask,
}

impl WorldHeightmap {
    pub fn new(seed: u64) -> Self {
        Self {
            plains: PlainsHeightmap::new(seed),
            continent: ContinentMask::new(seed),
        }
    }

    pub fn with_configs(plains: PlainsConfig, continent: ContinentConfig) -> Self {
        Self {
            plains: PlainsHeightmap::with_config(plains),
            continent: ContinentMask::with_config(continent),
        }
    }

    pub fn sea_level(&self) -> f64 {
        SEA_LEVEL
    }

    /// 权威采样: i64 方块坐标 → 高度 (格, f64).
    pub fn height(&self, x: i64, z: i64) -> f64 {
        self.height_f64(x as f64, z as f64)
    }

    fn height_f64(&self, x: f64, z: f64) -> f64 {
        let cc = self.continent.config();
        let c = self.continent.field(x, z);
        let land = smoothstep(cc.c_lo, cc.c_hi, c);
        let deep = 1.0 - smoothstep(cc.deep_lo, cc.deep_hi, c);
        let seabed = self.continent.seabed.get_noise_2d(x, z) as f64;
        let floor = SEA_LEVEL - cc.shelf_depth - cc.abyss_depth * deep
            + cc.seabed_amp * seabed * (1.0 - land);
        let land_h = self.plains.height_f64_pub(x, z);
        let shore = 1.0 - land;
        land_h - (land_h - floor) * shore.powf(cc.cliff_pow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let w = WorldHeightmap::new(1234);
        assert_eq!(w.height(0, 0), w.height(0, 0));
        assert_eq!(
            w.height(-5_000_000, 3_000_000),
            w.height(-5_000_000, 3_000_000)
        );
    }

    #[test]
    fn land_and_sea_both_exist() {
        // 32768 格范围粗扫: 海陆必须同时存在.
        let w = WorldHeightmap::new(ContinentConfig::default().seed);
        let (mut land, mut sea) = (0u32, 0u32);
        let mut i = -16384i64;
        while i <= 16384 {
            let mut j = -16384i64;
            while j <= 16384 {
                if w.height(i, j) > SEA_LEVEL {
                    land += 1;
                } else {
                    sea += 1;
                }
                j += 512;
            }
            i += 512;
        }
        assert!(
            land > 0 && sea > 0,
            "海陆必须同时存在: land={land} sea={sea}"
        );
        let land_frac = land as f64 / (land + sea) as f64;
        assert!(
            (0.25..0.80).contains(&land_frac),
            "陆地占比离谱: {land_frac:.2}"
        );
    }

    #[test]
    fn continuous_no_teleport() {
        // 相邻列高差有界 (悬崖处也不许跳变; 幂曲线单调, 单格落差 < 6).
        let w = WorldHeightmap::new(99);
        let mut worst = 0.0f64;
        for i in 0..4096 {
            let (x, z) = (i * 7 - 20000, i * 13 + 5000);
            let h = w.height(x, z);
            assert!(h.is_finite());
            for (dx, dz) in [(1, 0), (0, 1)] {
                let d = (w.height(x + dx, z + dz) - h).abs();
                worst = worst.max(d);
                assert!(d < 6.0, "跳变过大 ({x},{z}): {d}");
            }
        }
        // 记录最坏情况供调参参考 (不做断言, 避免种子相关脆弱).
        eprintln!("worst adjacent step: {worst:.2}");
    }

    #[test]
    fn deep_ocean_floor_sane() {
        // 远海心 (c 很负) 应接近 SEA-16 上下.
        let w = WorldHeightmap::new(ContinentConfig::default().seed);
        // 找一块 c < deep_lo 的点: 粗扫定位.
        let target = ContinentConfig::default().deep_lo;
        let mut found = None;
        let mut i = -65536i64;
        while i <= 65536 && found.is_none() {
            let mut j = -65536i64;
            while j <= 65536 && found.is_none() {
                if w.continent.field(i as f64, j as f64) < target {
                    found = Some((i, j));
                }
                j += 1024;
            }
            i += 1024;
        }
        let (x, z) = found.expect("应存在深海区");
        let h = w.height(x, z);
        assert!(
            (SEA_LEVEL - 20.0..SEA_LEVEL - 10.0).contains(&h),
            "深海底高程离谱 ({x},{z}): {h}"
        );
    }

    #[test]
    fn far_coords_finite() {
        let w = WorldHeightmap::new(7);
        for (x, z) in [
            (30_000_000i64, 0),
            (-30_000_000, 10_000_000),
            (0, -30_000_000),
        ] {
            let h = w.height(x, z);
            assert!(h.is_finite(), "远坐标非有限 ({x},{z})");
        }
    }
}
