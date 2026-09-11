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

use super::plains::{
    PlainsConfig, PlainsHeightmap, make_noise, seeded_coordinate_offset, smoothstep,
};

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
    sample_offset_x: f64,
    sample_offset_z: f64,
    field: FastNoiseLite,
    shore: FastNoiseLite,
    seabed: FastNoiseLite,
    warp_x: FastNoiseLite,
    warp_z: FastNoiseLite,
    /// 低频地貌控制场：正值区域才长出大陆内部山系。
    mountain: FastNoiseLite,
    /// 细长脊线场，取绝对值的反相后形成阿尔卑斯式山脊。
    ridge: FastNoiseLite,
    /// 侵蚀场：高侵蚀区降低山体，低侵蚀区保留陡峭峰面。
    erosion: FastNoiseLite,
    /// 河流中心线场。绝对值接近零的位置是连续的河谷。
    river: FastNoiseLite,
    river_detail: FastNoiseLite,
}

impl ContinentMask {
    pub fn new(seed: u64) -> Self {
        Self::with_config(ContinentConfig {
            seed,
            ..ContinentConfig::default()
        })
    }

    pub fn with_config(config: ContinentConfig) -> Self {
        let sample_offset_x = seeded_coordinate_offset(config.seed, 0x43_4f_4e_54_49_4e_01);
        let sample_offset_z = seeded_coordinate_offset(config.seed, 0x43_4f_4e_54_49_4e_02);
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
        let mountain = make_noise(
            next_seed(),
            NoiseType::OpenSimplex2,
            FractalType::FBm,
            3,
            1.0 / 9600.0,
        );
        let ridge = make_noise(
            next_seed(),
            NoiseType::OpenSimplex2,
            FractalType::FBm,
            3,
            1.0 / 2600.0,
        );
        let erosion = make_noise(
            next_seed(),
            NoiseType::OpenSimplex2,
            FractalType::FBm,
            2,
            1.0 / 1800.0,
        );
        let river = make_noise(
            next_seed(),
            NoiseType::OpenSimplex2,
            FractalType::FBm,
            2,
            1.0 / 1900.0,
        );
        let river_detail = make_noise(
            next_seed(),
            NoiseType::ValueCubic,
            FractalType::FBm,
            2,
            1.0 / 420.0,
        );
        Self {
            config,
            sample_offset_x,
            sample_offset_z,
            field,
            shore,
            seabed,
            warp_x,
            warp_z,
            mountain,
            ridge,
            erosion,
            river,
            river_detail,
        }
    }

    pub fn config(&self) -> &ContinentConfig {
        &self.config
    }

    /// 海陆场 c ∈ [-1, 1]. >0 倾向陆, <0 倾向海. 纯函数, f64 全程.
    pub fn field(&self, x: f64, z: f64) -> f64 {
        let c = &self.config;
        let (sx, sz) = self.sample_coords(x, z);
        let wx = sx + c.warp_amp * self.warp_x.get_noise_2d(sx, sz) as f64;
        let wz = sz + c.warp_amp * self.warp_z.get_noise_2d(sx, sz) as f64;
        self.field.get_noise_2d(wx, wz) as f64
            + c.shore_amp * self.shore.get_noise_2d(sx, sz) as f64
    }

    /// 陆地权重 0--1.
    pub fn land_factor(&self, x: f64, z: f64) -> f64 {
        let c = &self.config;
        smoothstep(c.c_lo, c.c_hi, self.field(x, z))
    }

    /// 山系权重、脊线权重和侵蚀度，均为 0--1 的可组合场。
    pub fn mountain_profile(&self, x: f64, z: f64) -> (f64, f64, f64) {
        let (x, z) = self.sample_coords(x, z);
        // 让山系集中在更大、更少的高值区；低值和中性区保留为
        // 连续平原。较低的旧门限 (-0.18) 会让轻微起伏几乎遍布陆地。
        let mountain = smoothstep(0.02, 0.38, self.mountain.get_noise_2d(x, z) as f64);
        let ridge_noise = self.ridge.get_noise_2d(x, z) as f64;
        let ridge = (1.0 - ridge_noise.abs()).powi(2);
        let erosion = 0.5 + 0.5 * self.erosion.get_noise_2d(x, z) as f64;
        (mountain, ridge, erosion)
    }

    /// 河流强度和未切割河岸高度。
    ///
    /// 绝对值零线的做法与参考数据包的 ridge/water 分支相同：河网天然
    /// 连续，不会像独立的随机圆斑那样断成一串池塘。
    pub fn river_profile(&self, x: f64, z: f64) -> (f64, f64) {
        let (x, z) = self.sample_coords(x, z);
        let center = self.river.get_noise_2d(x, z) as f64;
        let detail = self.river_detail.get_noise_2d(x, z) as f64;
        let distance = (center.abs() + detail.abs() * 0.035).clamp(0.0, 1.0);
        let channel = 1.0 - smoothstep(0.018, 0.095, distance);
        let bank = 1.0 - smoothstep(0.06, 0.22, distance);
        (channel, bank)
    }

    fn sample_coords(&self, x: f64, z: f64) -> (f64, f64) {
        (x + self.sample_offset_x, z + self.sample_offset_z)
    }
}

/// 一次完整的二维地形采样，供高度图和柱生成器共享，确保水系/山地边界一致。
#[derive(Clone, Copy, Debug)]
pub struct TerrainSample {
    pub height: f64,
    pub base_height: f64,
    pub land_factor: f64,
    pub mountain_factor: f64,
    pub ridge_factor: f64,
    pub erosion: f64,
    pub river_factor: f64,
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

    pub fn sample(&self, x: i64, z: i64) -> TerrainSample {
        self.sample_f64(x as f64, z as f64)
    }

    /// 权威采样: i64 方块坐标 → 高度 (格, f64).
    pub fn height(&self, x: i64, z: i64) -> f64 {
        self.sample(x, z).height
    }

    fn sample_f64(&self, x: f64, z: f64) -> TerrainSample {
        let cc = self.continent.config();
        let c = self.continent.field(x, z);
        let land = smoothstep(cc.c_lo, cc.c_hi, c);
        let deep = 1.0 - smoothstep(cc.deep_lo, cc.deep_hi, c);
        let (sample_x, sample_z) = self.continent.sample_coords(x, z);
        let seabed = self.continent.seabed.get_noise_2d(sample_x, sample_z) as f64;
        let floor = SEA_LEVEL - cc.shelf_depth - cc.abyss_depth * deep
            + cc.seabed_amp * seabed * (1.0 - land);
        let plains_h = self.plains.height_f64_pub(x, z);
        let (mountain, ridge, erosion) = self.continent.mountain_profile(x, z);
        // 山系首先由超低频 mask 决定，再由脊线与侵蚀共同塑形。
        // 峰顶控制在世界上限以下，为树木、结构和玩家留出空间。
        let mountain_h = mountain * (7.0 + 38.0 * ridge) * (0.42 + 0.58 * (1.0 - erosion));
        let base_height = plains_h + land * mountain_h;
        let (river_channel, river_bank) = self.continent.river_profile(x, z);
        // 河流只在陆地内部切割；高山地区的河谷变窄，山脚则更宽。
        let mountain_suppression = 1.0 - mountain * 0.62;
        let river_factor = land * river_channel * mountain_suppression;
        let river_carve = river_factor * (2.5 + 5.0 * river_bank);
        let land_h = base_height - river_carve;
        let shore = 1.0 - land;
        let height = land_h - (land_h - floor) * shore.powf(cc.cliff_pow);
        TerrainSample {
            height,
            base_height,
            land_factor: land,
            mountain_factor: mountain,
            ridge_factor: ridge,
            erosion,
            river_factor,
        }
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
    fn world_origin_varies_with_seed() {
        let mut land_origins = 0;
        let mut sea_origins = 0;
        for seed in 0..32 {
            let w = WorldHeightmap::new(seed);
            if w.continent.land_factor(0.0, 0.0) >= 0.5 {
                land_origins += 1;
            } else {
                sea_origins += 1;
            }
        }
        assert!(
            land_origins > 0 && sea_origins > 0,
            "世界原点应随种子分布在海陆两侧: land={land_origins} sea={sea_origins}"
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

    #[test]
    fn macro_profile_has_mountains_and_rivers() {
        let w = WorldHeightmap::new(2026_0904);
        let mut highest = f64::NEG_INFINITY;
        let mut river_columns = 0;
        for x in (-12000..=12000).step_by(128) {
            for z in (-12000..=12000).step_by(128) {
                let sample = w.sample(x, z);
                highest = highest.max(sample.height);
                if sample.river_factor > 0.55 && sample.land_factor > 0.8 {
                    river_columns += 1;
                }
            }
        }
        assert!(highest > SEA_LEVEL + 12.0, "应出现明显山地: {highest}");
        assert!(river_columns > 10, "应出现连续河谷采样点: {river_columns}");
    }

    #[test]
    fn plains_have_broad_contiguous_runs() {
        let w = WorldHeightmap::new(2026_0904);
        let mut longest_run = 0;
        for z in (-16000..=16000).step_by(256) {
            let mut run = 0;
            for x in (-16000..=16000).step_by(256) {
                let sample = w.sample(x, z);
                if sample.land_factor >= 0.95 && sample.mountain_factor <= 0.22 {
                    run += 1;
                    longest_run = longest_run.max(run);
                } else {
                    run = 0;
                }
            }
        }
        assert!(
            longest_run >= 8,
            "应存在至少 2048 格宽的连续平原，当前最长采样串为 {longest_run}"
        );
    }
}
