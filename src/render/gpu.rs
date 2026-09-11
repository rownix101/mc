//! GPU 初始化与 3D 体素渲染。
//!
//! 这一层只负责资源和 draw call；世界查询、面剔除和网格生命周期由上层处理。

use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use winit::dpi::PhysicalSize;
use winit::window::Window;

use super::gpu_mesh::{DrawBatch, GpuVertex, mesh_batches, visible_ranges};
use super::probe::{self, ProbeVolume};
#[cfg(test)]
use super::voxel::Vertex;
use super::voxel::VoxelMesh;
use crate::world::atlas::{self, Atlas};

const MENU_CLEAR: wgpu::Color = wgpu::Color {
    r: 0.05,
    g: 0.07,
    b: 0.12,
    a: 1.0,
};

pub const WORLD_CLEAR: wgpu::Color = wgpu::Color {
    r: 0.38,
    g: 0.62,
    b: 0.84,
    a: 1.0,
};

/// 原版 Minecraft 的完整昼夜循环为 24,000 游戏刻，即 20 分钟现实时间。
const VANILLA_DAY_LENGTH_SECONDS: f32 = 20.0 * 60.0;
const SUN_ANGULAR_SPEED: f32 = std::f32::consts::TAU / VANILLA_DAY_LENGTH_SECONDS;
const SHADOW_SIZE: u32 = 2048;
const SHADOW_RADIUS: f32 = 96.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    /// 从被照表面指向太阳的方向。
    sun_direction: [f32; 4],
    /// RGB 为太阳颜色，A 为太阳强度。
    sun_color: [f32; 4],
    /// RGB 为天顶颜色，A 为环境光强度。
    sky_top: [f32; 4],
    /// RGB 为地平线颜色，A 为环境光强度。
    sky_horizon: [f32; 4],
    /// RGB 为雾色，A 为雾密度。
    fog_color: [f32; 4],
    /// x = 白昼比例；y/z/w = atlas 列数、归一化 stride、归一化 gutter。
    params: [f32; 4],
    /// 当前 surface 的像素尺寸，用于从 fullscreen sky 三角形重建视线。
    viewport: [f32; 4],
    /// 游戏运行时间（秒），供水面波纹使用。
    time: [f32; 4],
    /// camera-relative 网格原点；水面波形使用它保持世界坐标稳定。
    world_origin: [f32; 4],
    /// camera 在当前 camera-relative 网格中的位置。
    camera_position: [f32; 4],
    shadow_view_proj: [[f32; 4]; 4],
    /// 探针网格最小角的 camera-relative 坐标 (xyz); w 保留。
    probe_origin: [f32; 4],
    /// xyz = 1/(spacing*dims) 的 UVW 缩放; w > 0.5 表示探针可用。
    probe_params: [f32; 4],
}

const SKY_SHADER: &str = r#"
struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    sky_top: vec4<f32>,
    sky_horizon: vec4<f32>,
    fog_color: vec4<f32>,
    params: vec4<f32>,
    viewport: vec4<f32>,
    time: vec4<f32>,
    world_origin: vec4<f32>,
    camera_position: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> frame: Frame;

struct SkyOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ray: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> SkyOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let clip_xy = positions[index];
    let far_point = frame.inv_view_proj * vec4<f32>(clip_xy, 1.0, 1.0);
    var output: SkyOutput;
    output.clip_position = vec4<f32>(clip_xy, 1.0, 1.0);
    output.ray = far_point.xyz / far_point.w - frame.camera_position.xyz;
    return output;
}

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 456.21));
    q = q + vec2<f32>(dot(q, q.yx + vec2<f32>(45.32)));
    return fract(q.x * q.y);
}

// Smooth 3D density gives clouds a shaded interior rather than two painted sheets.
fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let q = i.xy + i.z * vec2<f32>(37.0, 113.0);
    let offset = vec2<f32>(37.0, 113.0);
    let a = mix(mix(hash21(q), hash21(q + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash21(q + vec2<f32>(0.0, 1.0)), hash21(q + vec2<f32>(1.0)), u.x), u.y);
    let b = mix(mix(hash21(q + offset), hash21(q + offset + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash21(q + offset + vec2<f32>(0.0, 1.0)), hash21(q + offset + vec2<f32>(1.0)), u.x), u.y);
    return mix(a, b, u.z);
}

// Each weather cell contains an irregular cluster of rounded updrafts.
// Clusters fit within their cell, so crossing a cell boundary stays empty and continuous.
fn cloud_distance(p: vec3<f32>) -> f32 {
    let cell = floor(p.xz / 1300.0);
    let seed = hash21(cell);
    if (seed < 0.48) { return 300.0; }
    let center = (cell + vec2<f32>(0.5)) * 1300.0
        + vec2<f32>(hash21(cell + 13.0), hash21(cell + 71.0)) * 400.0 - 200.0;
    let local = p - vec3<f32>(center.x, 620.0 + hash21(cell + 51.0) * 180.0, center.y);
    let angle = seed * 31.0;
    let q = vec3<f32>(local.x * cos(angle) - local.z * sin(angle), local.y,
        local.x * sin(angle) + local.z * cos(angle));
    let radius = 130.0 + hash21(cell + 29.0) * 120.0;
    var d = length(q / vec3<f32>(1.35, 0.72, 1.0)) - radius;
    d = min(d, length(q - vec3<f32>(-110.0, 70.0, 20.0)) - radius * 0.76);
    d = min(d, length(q - vec3<f32>(85.0, 115.0, -35.0)) - radius * 0.82);
    d = min(d, length(q - vec3<f32>(10.0, 110.0 + hash21(cell + 7.0) * 120.0, 60.0)) - radius * 0.65);
    d = min(d, length(q - vec3<f32>(150.0, 20.0, 75.0)) - radius * 0.65);
    // Only evaluate turbulent detail near the cloud. It breaks up the lobes without
    // blurring the whole silhouette into translucent noise.
    if (d < 65.0) {
        let detail = noise3(p * 0.018) * 0.60 + noise3(p * 0.041) * 0.28
            + noise3(p * 0.089) * 0.12;
        d = d + (detail - 0.42) * 95.0;
    }
    return d;
}

fn cloud_density(p: vec3<f32>) -> f32 {
    return clamp(-cloud_distance(p) / 32.0, 0.0, 1.0);
}

@fragment
fn fs_main(input: SkyOutput) -> @location(0) vec4<f32> {
    let ray = normalize(input.ray);
    let day = clamp(frame.params.x, 0.0, 1.0);
    let night = 1.0 - day;
    let sun = normalize(frame.sun_direction.xyz);
    let mu = dot(ray, sun);
    let altitude = 1.0 - exp(-max(ray.y, 0.0) * 3.0);
    var sky = mix(frame.sky_horizon.rgb, frame.sky_top.rgb, altitude);
    if (frame.time.y > 0.5) {
        return vec4<f32>(mix(vec3<f32>(0.025, 0.24, 0.30),
            vec3<f32>(0.08, 0.48, 0.54), altitude * 0.45), 1.0);
    }

    // Directional haze: a broad aureole, with sunset colour confined to the sunward horizon.
    let twilight = (1.0 - smoothstep(0.0, 0.40, abs(sun.y))) * day;
    let horizon = exp(-max(ray.y, 0.0) * 8.0);
    let forward = pow(max(mu, 0.0), 8.0);
    sky = sky + frame.sun_color.rgb * forward * (0.045 * day);
    sky = mix(sky, vec3<f32>(0.72, 0.26, 0.09),
        twilight * horizon * pow(max(mu, 0.0), 3.0) * 0.65);
    // Blend the lower hemisphere to the same colour used by terrain aerial perspective.
    sky = mix(sky, frame.fog_color.rgb, 1.0 - smoothstep(-0.12, 0.015, ray.y));

    // Angular discs with screen-space antialiasing instead of Gaussian pinpoints.
    let sun_distance = length(ray - sun);
    let moon_distance = length(ray + sun);
    let sun_aa = max(fwidth(sun_distance), 0.00015);
    let moon_aa = max(fwidth(moon_distance), 0.00015);
    let sun_disc = 1.0 - smoothstep(0.00465 - sun_aa, 0.00465 + sun_aa, sun_distance);
    let moon_disc = 1.0 - smoothstep(0.0045 - moon_aa, 0.0045 + moon_aa, moon_distance);
    sky = sky + frame.sun_color.rgb * (sun_disc * 5.0 + pow(max(mu, 0.0), 256.0) * 0.12)
        * smoothstep(-0.015, 0.035, sun.y);
    sky = sky + vec3<f32>(0.55, 0.64, 0.80) * moon_disc * night
        * smoothstep(-0.015, 0.035, -sun.y);

    let star_cell = ray * 230.0;
    let cell = floor(star_cell);
    let local = fract(star_cell) - vec3<f32>(0.5);
    let seed = hash21(cell.xy + cell.z * 0.37);
    let star = smoothstep(0.995, 1.0, seed) * (1.0 - smoothstep(0.06, 0.22, length(local)));
    sky = sky + vec3<f32>(0.56, 0.65, 0.85) * star * night * night
        * smoothstep(0.0, 0.2, ray.y);

    // Adaptive traversal through tall cumulus volumes. Empty space takes large
    // steps, while cloud boundaries and interiors are integrated at 14-block spacing.
    let eye = frame.world_origin.xyz + frame.camera_position.xyz;
    if (ray.y > 0.015 && eye.y < 1280.0) {
        var distance = max((400.0 - eye.y) / ray.y, 0.0);
        let end = min((1280.0 - eye.y) / ray.y, 7000.0);
        let wind = vec3<f32>(frame.time.x * 1.8, 0.0, frame.time.x * 0.65);
        var transmission = 1.0;
        var cloud_light = vec3<f32>(0.0);
        for (var i = 0; i < 96; i = i + 1) {
            if (distance >= end || transmission < 0.015) { break; }
            let p = eye + ray * distance + wind;
            let boundary = cloud_distance(p);
            let step_length = clamp(boundary * 0.65, 14.0, 180.0);
            let density = clamp(-boundary / 32.0, 0.0, 1.0);
            if (density > 0.001) {
                let opacity = 1.0 - exp(-density * step_length * 0.065);
                let light_depth = cloud_density(p + sun * 35.0) * 35.0
                    + cloud_density(p + sun * 90.0) * 55.0
                    + cloud_density(p + sun * 180.0) * 90.0;
                let sunlight = exp(-light_depth * 0.025);
                let height_light = smoothstep(470.0, 950.0, p.y);
                let ambient = mix(vec3<f32>(0.006, 0.009, 0.018),
                    mix(vec3<f32>(0.17, 0.22, 0.30), vec3<f32>(0.38, 0.43, 0.50), height_light), day);
                let direct = frame.sun_color.rgb * sunlight * day * 0.90;
                let lining = frame.sun_color.rgb * pow(max(mu, 0.0), 12.0) * sunlight * day * 0.18;
                let lit = mix(ambient + direct + lining, frame.sky_horizon.rgb,
                    1.0 - exp(-distance * 0.00032));
                cloud_light = cloud_light + transmission * opacity * lit;
                transmission = transmission * (1.0 - opacity);
            }
            distance = distance + step_length;
        }
        sky = sky * transmission + cloud_light;
    }
    return vec4<f32>(max(sky, vec3<f32>(0.0)), 1.0);
}

"#;

const VOXEL_SHADER: &str = r#"
struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    sky_top: vec4<f32>,
    sky_horizon: vec4<f32>,
    fog_color: vec4<f32>,
    params: vec4<f32>,
    viewport: vec4<f32>,
    time: vec4<f32>,
    world_origin: vec4<f32>,
    camera_position: vec4<f32>,
    shadow_view_proj: mat4x4<f32>,
    probe_origin: vec4<f32>,
    probe_params: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> frame: Frame;

@group(0) @binding(1)
var atlas_texture: texture_2d<f32>;

@group(0) @binding(2)
var atlas_normal: texture_2d<f32>;

@group(0) @binding(3)
var atlas_roughness: texture_2d<f32>;

@group(0) @binding(4)
var atlas_sampler: sampler;

// 世界空间辐照度探针: SH L1 天空可见度.
@group(0) @binding(5)
var probe_volume: texture_3d<f32>;

@group(0) @binding(6)
var probe_sampler: sampler;

@group(1) @binding(0)
var scene_texture: texture_2d<f32>;

@group(1) @binding(1)
var scene_sampler: sampler;

@group(2) @binding(0) var shadow_map: texture_depth_2d;
@group(2) @binding(1) var shadow_sampler: sampler_comparison;

// Comparison filtering gives a soft 3x3 PCF footprint. Fade before the map
// boundary so clamped samples never cast a stripe outside the covered region.
fn sun_visibility(position: vec3<f32>, normal: vec3<f32>) -> f32 {
    let slope = 1.0 - max(dot(normal, frame.sun_direction.xyz), 0.0);
    let clip = frame.shadow_view_proj * vec4<f32>(position + normal * (0.015 + 0.045 * slope), 1.0);
    let uv = clip.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    if clip.z <= 0.0 || clip.z >= 1.0 || any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
        return 1.0;
    }
    let texel = 1.0 / vec2<f32>(textureDimensions(shadow_map));
    // The nine bilinear comparisons form a separable four-texel footprint
    // with weights (1-s, 1, 1, s). Pair adjacent texels on each axis:
    // four weighted bilinear taps reproduce the original 3x3 PCF kernel.
    let phase = fract(uv / texel - vec2<f32>(0.5));
    let w0 = vec2<f32>(2.0) - phase;
    let w1 = vec2<f32>(1.0) + phase;
    let o0 = -vec2<f32>(1.0) + vec2<f32>(1.0) / w0 - phase;
    let o1 = vec2<f32>(1.0) + phase / w1 - phase;
    let depth = clip.z - 0.00004;
    let visibility =
        textureSampleCompareLevel(shadow_map, shadow_sampler, uv + o0 * texel, depth) * w0.x * w0.y
        + textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>(o1.x, o0.y) * texel, depth) * w1.x * w0.y
        + textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>(o0.x, o1.y) * texel, depth) * w0.x * w1.y
        + textureSampleCompareLevel(shadow_map, shadow_sampler, uv + o1 * texel, depth) * w1.x * w1.y;
    let edge = max(abs(uv.x * 2.0 - 1.0), abs(uv.y * 2.0 - 1.0));
    return mix(visibility / 9.0, 1.0, smoothstep(0.82, 0.98, edge));
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) ao: f32,
    @location(4) material: f32,
    @location(5) tangent: vec4<f32>,
    @location(6) bitangent: vec4<f32>,
    @location(7) local_uv: vec2<f32>,
    @location(8) blend_tiles: vec4<u32>,
    @location(9) texture_region: vec4<u32>,
    @location(10) light: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) ao: f32,
    @location(3) material: f32,
    @location(4) world_position: vec3<f32>,
    @location(5) tangent: vec3<f32>,
    @location(6) bitangent: vec3<f32>,
    @location(7) local_uv: vec2<f32>,
    @location(8) @interpolate(flat) blend_tiles: vec4<u32>,
    @location(9) @interpolate(flat) texture_region: vec4<u32>,
    @location(10) relative_position: vec3<f32>,
    @location(11) sky_light: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    var position = input.position;
    let is_water = abs(input.material - 1.0) < 0.5;
    let world_xz = input.position.xz + frame.world_origin.xz;
    if is_water && input.normal.y > 0.5 {
        position.y = position.y + water_height(world_xz, frame.time.x);
    }
    output.clip_position = frame.view_proj * vec4<f32>(position, 1.0);
    output.uv = input.uv;
    output.normal = input.normal.xyz;
    output.ao = input.ao;
    output.material = input.material;
    output.world_position = position + frame.world_origin.xyz;
    output.relative_position = position;
    output.tangent = input.tangent.xyz;
    output.bitangent = input.bitangent.xyz;
    output.local_uv = input.local_uv;
    output.blend_tiles = input.blend_tiles;
    output.texture_region = input.texture_region;
    output.sky_light = input.light;
    return output;
}

struct ShadowOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) material: f32,
};

@vertex
fn vs_shadow(input: VertexInput) -> ShadowOutput {
    var output: ShadowOutput;
    output.position = frame.shadow_view_proj * vec4<f32>(input.position, 1.0);
    output.uv = input.uv;
    output.material = input.material;
    return output;
}

@fragment
fn fs_shadow(input: ShadowOutput) {
    // Only water is excluded from the shadow map; leaves keep their alpha
    // cutout so gaps between leaves let dappled sunlight through.
    if textureSample(atlas_texture, atlas_sampler, input.uv).a < 0.5
        || abs(input.material - 1.0) < 0.5
    {
        discard;
    }
}

// Four cheap, phase-shifted Gerstner-like components reproduce Photon’s
// layered water motion without requiring a noise texture or a tessellated
// surface.  The absolute world coordinate is important: the mesh origin is
// periodically recentered around the player.
fn water_height(coord: vec2<f32>, time: f32) -> f32 {
    let d0 = normalize(vec2<f32>(0.866, 0.500));
    let d1 = normalize(vec2<f32>(-0.342, 0.940));
    let d2 = normalize(vec2<f32>(0.966, -0.259));
    let d3 = normalize(vec2<f32>(-0.707, -0.707));
    let w0 = sin(dot(coord, d0) * 2.1 - time * 1.05) * 0.018;
    let w1 = sin(dot(coord, d1) * 4.8 + time * 1.55 + 1.7) * 0.010;
    let w2 = sin(dot(coord, d2) * 8.7 - time * 2.20 + 4.1) * 0.005;
    let w3 = sin(dot(coord, d3) * 14.0 + time * 2.75 + 2.4) * 0.003;
    return w0 + w1 + w2 + w3;
}

fn water_normal(world_position: vec3<f32>, flat_normal: vec3<f32>, time: f32) -> vec3<f32> {
    if flat_normal.y < 0.5 {
        return flat_normal;
    }
    // sin(a+b)-sin(a-b) = 2*cos(a)*sin(b). These coefficients
    // preserve the original h=0.08 central difference, including smoothing.
    let p = world_position.xz;
    var gradient = vec2<f32>(0.0);
    gradient += cos(dot(p, normalize(vec2<f32>(0.866, 0.500))) * 2.1
        + time * -1.05 + 0.0) * vec2<f32>(0.032620153, 0.018878196);
    gradient += cos(dot(p, normalize(vec2<f32>(-0.342, 0.940))) * 4.8
        + time * 1.55 + 1.7) * vec2<f32>(-0.016364265, 0.044134667);
    gradient += cos(dot(p, normalize(vec2<f32>(0.966, -0.259))) * 8.7
        + time * -2.20 + 4.1) * vec2<f32>(0.038922061, -0.011204268);
    gradient += cos(dot(p, normalize(vec2<f32>(-0.707, -0.707))) * 14.0
        + time * 2.75 + 2.4) * vec2<f32>(-0.026689918, -0.026689918);
    return normalize(vec3<f32>(-gradient.x, 1.0, -gradient.y));
}

fn packed_atlas_uv(tile: u32, local_uv: vec2<f32>, region: vec4<u32>) -> vec2<f32> {
    let atlas_columns = u32(frame.params.y);
    let stride_uv = frame.params.z;
    let gutter_uv = frame.params.w;
    let tile_uv = stride_uv - gutter_uv * 2.0;
    let cell = vec2<f32>(f32(tile % atlas_columns), f32(tile / atlas_columns));
    let region_count = vec2<f32>(f32(region.z), f32(region.w));
    let region_size = vec2<f32>(tile_uv) / region_count;
    return cell * stride_uv + vec2<f32>(gutter_uv)
        + (vec2<f32>(f32(region.x), f32(region.y)) + local_uv) * region_size;
}

fn transition_noise(position: vec3<f32>) -> f32 {
    let cell = floor(position * 7.0);
    return fract(sin(dot(cell, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453) * 2.0 - 1.0;
}

// Overlapping world-anchored patches sample only the interior of the image.
// No wrapping or random rotations: even imperfectly tileable source maps stay
// continuous, and every normal remains in the same tangent frame.
struct SandSample {
    color: vec4<f32>,
    normal: vec4<f32>,
    roughness: f32,
};

fn sand_sample(tile: u32, p: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> SandSample {
    let base = floor(p);
    let f = fract(p);
    let fade = f * f * f * (f * (f * 6.0 - vec2<f32>(15.0)) + vec2<f32>(10.0));
    let tile_size = frame.params.z - 2.0 * frame.params.w;
    var result: SandSample;
    result.color = vec4<f32>(0.0);
    result.normal = vec4<f32>(0.0);
    result.roughness = 0.0;
    for (var y = 0u; y < 2u; y++) {
        for (var x = 0u; x < 2u; x++) {
            let cell = base + vec2<f32>(f32(x), f32(y));
            // Integer hash avoids large sine arguments and is stable across chunks.
            let c = vec2<u32>(vec2<i32>(cell));
            var h = c.x * 1664525u + c.y * 1013904223u;
            h = (h ^ (h >> 16u)) * 2246822519u;
            h = h ^ (h >> 13u);
            let offset = vec2<f32>(f32(h & 65535u), f32(h >> 16u)) / 65535.0;
            let local = vec2<f32>(0.3) + offset * 0.4 + (p - cell) * 0.25;
            let uv = packed_atlas_uv(tile, local, vec4<u32>(0u, 0u, 1u, 1u));
            let w = select(1.0 - fade.x, fade.x, x == 1u)
                * select(1.0 - fade.y, fade.y, y == 1u);
            result.color += textureSampleGrad(atlas_texture, atlas_sampler, uv, dx * tile_size * 0.25, dy * tile_size * 0.25) * w;
            result.normal += textureSampleGrad(atlas_normal, atlas_sampler, uv, dx * tile_size * 0.25, dy * tile_size * 0.25) * w;
            result.roughness += textureSampleGrad(atlas_roughness, atlas_sampler, uv, dx * tile_size * 0.25, dy * tile_size * 0.25).r * w;
        }
    }
    return result;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var texel = textureSample(atlas_texture, atlas_sampler, input.uv);
    var normal_texel = textureSample(atlas_normal, atlas_sampler, input.uv);
    var roughness = textureSample(atlas_roughness, atlas_sampler, input.uv).r;
    let jitter = transition_noise(input.world_position) * 0.035;
    let blend_width = 0.16;
    var weights = vec4<f32>(
        1.0 - smoothstep(0.0, blend_width, input.local_uv.x + jitter),
        1.0 - smoothstep(0.0, blend_width, 1.0 - input.local_uv.x - jitter),
        1.0 - smoothstep(0.0, blend_width, input.local_uv.y + jitter),
        1.0 - smoothstep(0.0, blend_width, 1.0 - input.local_uv.y - jitter)
    );
    let atlas_cell = vec2<u32>(floor(input.uv / frame.params.z));
    let current_tile = atlas_cell.y * u32(frame.params.y) + atlas_cell.x;
    let sand_position = vec2<f32>(dot(input.world_position, input.tangent),
        dot(input.world_position, input.bitangent)) * 2.0;
    let sand_dx = dpdx(sand_position);
    let sand_dy = dpdy(sand_position);
    if abs(input.material - 4.0) < 0.5 {
        let sand = sand_sample(current_tile, sand_position, sand_dx, sand_dy);
        texel = sand.color;
        normal_texel = sand.normal;
        roughness = sand.roughness;
    }

    weights *= vec4<f32>(
        select(0.0, 1.0, input.blend_tiles.x != current_tile),
        select(0.0, 1.0, input.blend_tiles.y != current_tile),
        select(0.0, 1.0, input.blend_tiles.z != current_tile),
        select(0.0, 1.0, input.blend_tiles.w != current_tile)
    );
    let region_size = vec2<f32>(frame.params.z - frame.params.w * 2.0)
        / vec2<f32>(input.texture_region.zw);
    let blend_dx = dpdx(input.local_uv) * region_size;
    let blend_dy = dpdy(input.local_uv) * region_size;
    let edge_weight = dot(weights, vec4<f32>(1.0));
    if edge_weight > 0.0001 {
        // Only sample neighbors that actually contribute. Explicit gradients
        // keep mip selection valid inside pixel-dependent control flow.
        for (var edge = 0u; edge < 4u; edge++) {
            if weights[edge] > 0.0 {
                let uv = packed_atlas_uv(input.blend_tiles[edge], input.local_uv, input.texture_region);
                texel += textureSampleGrad(atlas_texture, atlas_sampler, uv, blend_dx, blend_dy) * weights[edge];
                normal_texel += textureSampleGrad(atlas_normal, atlas_sampler, uv, blend_dx, blend_dy) * weights[edge];
                roughness += textureSampleGrad(atlas_roughness, atlas_sampler, uv, blend_dx, blend_dy).r * weights[edge];
            }
        }
        let denominator = 1.0 + edge_weight;
        texel /= denominator;
        normal_texel /= denominator;
        roughness /= denominator;
    }
    // Transparent texture pixels must not write depth. Glass is represented by
    // an opaque detail pattern on a transparent background; without this test,
    // the alpha-blended color is invisible but the depth value still hides the
    // terrain behind it, producing large clear-colored planes.
    if texel.a < 0.5 {
        discard;
    }
    let is_water = abs(input.material - 1.0) < 0.5;
    let is_foliage = abs(input.material - 2.0) < 0.5;
    let underwater = frame.time.y > 0.5;
    let visibility = sun_visibility(input.relative_position, input.normal);
    var color = texel.rgb;
    var alpha = texel.a;
    var normal = input.normal;
    if is_water {
        normal = water_normal(input.world_position, input.normal, frame.time.x);
        let view_dir = normalize(frame.camera_position.xyz
            - (input.world_position - frame.world_origin.xyz));
        let NoV = max(dot(normal, view_dir), 0.0);
        let fresnel = 0.035 + 0.965 * pow(1.0 - NoV, 5.0);
        let screen_uv = input.clip_position.xy / frame.viewport.xy;
        let distortion = normal.xz * (0.012 + 0.035 * fresnel);
        let refracted_uv = clamp(screen_uv + distortion, vec2<f32>(0.002), vec2<f32>(0.998));
        let reflected_uv = clamp(screen_uv - distortion * 1.8, vec2<f32>(0.002), vec2<f32>(0.998));
        let scene_refracted = textureSampleLevel(scene_texture, scene_sampler, refracted_uv, 0.0).rgb;
        let scene_reflected = textureSampleLevel(scene_texture, scene_sampler, reflected_uv, 0.0).rgb;

        // Beer-Lambert-like absorption: red disappears first, leaving the
        // characteristic teal/blue depth of Photon water.
        let distance = length(input.world_position - frame.world_origin.xyz
            - frame.camera_position.xyz);
        let water_distance = 1.4 + distance * 0.13;
        let transmittance = exp(-vec3<f32>(0.22, 0.065, 0.030) * water_distance);
        let scattering = vec3<f32>(0.010, 0.085, 0.125) * (1.0 - transmittance);
        let caustic = max(
            sin(input.world_position.x * 3.1 + frame.time.x * 1.3)
                * sin(input.world_position.z * 2.4 - frame.time.x * 1.0),
            0.0,
        );
        let sun_specular = pow(max(dot(reflect(-normalize(frame.sun_direction.xyz), normal), view_dir), 0.0), 72.0)
            * frame.sun_color.a * 1.4 * visibility;
        let surface = scene_refracted * transmittance + scattering;
        let reflection = scene_reflected * (0.22 + 0.48 * fresnel);
        color = surface * (1.0 - fresnel) + reflection;
        color = color + vec3<f32>(0.02, 0.11, 0.12) * caustic * (1.0 - fresnel) * visibility * frame.sun_color.a;
        color = color + frame.sun_color.rgb * sun_specular;
        color = color + vec3<f32>(0.025, 0.10, 0.11) * pow(max(normal.y, 0.0), 6.0);
        // This is a complete composite over the already rendered opaque
        // scene, so it intentionally replaces the sampled backdrop.
        alpha = 1.0;
    }
    if is_water {
        return vec4<f32>(max(color, vec3<f32>(0.0)), 1.0);
    }
    // World-space, continuous variation avoids a new grid at block boundaries.
    // Fine detail still comes from the mip-filtered normal/roughness atlas.
    let grass_top = abs(input.material - 3.0) < 0.5 && input.normal.y > 0.5;
    var tangent_normal = normal_texel.xyz * 2.0 - vec3<f32>(1.0);
    if grass_top {
        let p = input.world_position.xz;
        let grass_variation = sin(dot(p, vec2<f32>(0.19, 0.13)))
            * sin(dot(p, vec2<f32>(-0.09, 0.23)) + 1.7);
        color *= mix(vec3<f32>(0.88, 0.93, 0.82), vec3<f32>(1.08, 1.04, 0.96), grass_variation * 0.5 + 0.5);
        tangent_normal = normalize(vec3<f32>(tangent_normal.xy * 1.3, tangent_normal.z));
        roughness = clamp(roughness + grass_variation * 0.06, 0.58, 1.0);
    }
    normal = normalize(
        input.tangent * tangent_normal.x
        + input.bitangent * tangent_normal.y
        + input.normal * tangent_normal.z
    );
    if underwater && !is_water {
        let distance = length(input.world_position - frame.world_origin.xyz - frame.camera_position.xyz);
        let fog = clamp((distance - 2.5) / 28.0, 0.0, 1.0) * 0.72;
        let water_fog = vec3<f32>(0.035, 0.34, 0.40);
        color = mix(color, water_fog, fog);
        let caustic = max(
            sin(input.world_position.x * 2.1 + frame.time.x * 1.2)
                * sin(input.world_position.z * 1.7 - frame.time.x * 0.9),
            0.0,
        );
        color = color + vec3<f32>(0.05, 0.12, 0.10) * caustic * (1.0 - fog) * 0.8 * visibility * frame.sun_color.a;
    }
    normal = normalize(normal);
    let sun_direction = normalize(frame.sun_direction.xyz);
    let direct = max(dot(normal, sun_direction), 0.0)
        * frame.sun_color.a
        * frame.sun_color.rgb * visibility
        * step(0.0, dot(input.normal, sun_direction));
    let hemisphere = mix(frame.sky_horizon.rgb, frame.sky_top.rgb,
        clamp(normal.y * 0.5 + 0.5, 0.0, 1.0));
    // Directional probes retain sky orientation; propagated voxel light supplies
    // local fill under overhangs and guards against coarse probes leaking indoors.
    var sky_visibility = input.sky_light;
    if frame.probe_params.w > 0.5 {
        let probe_uvw =
            (input.relative_position - frame.probe_origin.xyz) * frame.probe_params.xyz;
        let probe_sh = textureSampleLevel(probe_volume, probe_sampler, probe_uvw, 0.0);
        let directional_sky = clamp(
            probe_sh.x * 0.2820948
                + (probe_sh.y * normal.y + probe_sh.z * normal.z + probe_sh.w * normal.x)
                    * 0.4886025,
            0.0,
            1.0,
        );
        // Fade to local light at the volume boundary instead of sampling a
        // clamped edge probe for all distant geometry.
        let edge = min(probe_uvw, vec3<f32>(1.0) - probe_uvw);
        let probe_weight = smoothstep(0.0, 0.08, min(edge.x, min(edge.y, edge.z)));
        let local_sky = mix(input.sky_light, min(directional_sky, input.sky_light), 0.65);
        sky_visibility = mix(input.sky_light, local_sky, probe_weight);
    }
    // Broad sky fill and a modest ground bounce favour sides/undersides.
    // Both vanish with local sky access; the enclosed-space floor is unchanged.
    let side_weight = 1.0 - max(input.normal.y, 0.0);
    let sky_fill = frame.sky_horizon.rgb * (0.28 * side_weight * input.sky_light);
    let ground_bounce = vec3<f32>(0.20, 0.17, 0.11)
        * frame.params.x * input.sky_light * (0.5 - 0.5 * input.normal.y);
    var ambient = (hemisphere * mix(0.15, 1.0, sky_visibility) + sky_fill + ground_bounce)
        * frame.sky_horizon.a;
    if underwater {
        ambient = vec3<f32>(0.04, 0.30, 0.36) * 0.72;
    }
    // Vertex AO belongs to indirect light; keeping direct sunlight separate avoids
    // making sunlit faces unnaturally black at voxel corners.
    var lit = ambient * input.ao;
    let view_direction = normalize(
        frame.camera_position.xyz - (input.world_position - frame.world_origin.xyz)
    );
    let half_direction = (sun_direction + view_direction)
        / max(length(sun_direction + view_direction), 0.0001);
    let NoV = max(dot(normal, view_direction), 0.001);
    let NoL = max(dot(normal, sun_direction), 0.0);
    let NoH = max(dot(normal, half_direction), 0.0);
    let VoH = max(dot(view_direction, half_direction), 0.0);
    // Cook-Torrance dielectric BRDF: GGX distribution, Smith visibility,
    // Schlick Fresnel. Atlas roughness is perceptual; square it for alpha.
    let a = max(roughness * roughness, 0.045);
    let a2 = a * a;
    let d = NoH * NoH * (a2 - 1.0) + 1.0;
    let distribution = a2 / (3.14159265 * d * d);
    let smith = 0.5 / max(NoL * sqrt(NoV * NoV * (1.0 - a2) + a2)
        + NoV * sqrt(NoL * NoL * (1.0 - a2) + a2), 0.0001);
    let fresnel = 0.04 + 0.96 * pow(1.0 - VoH, 5.0);
    // Sun intensity is calibrated as diffuse irradiance / PI.
    let specular = distribution * smith * fresnel * direct * 3.14159265;
    let distance = length(input.world_position - frame.world_origin.xyz - frame.camera_position.xyz);
    let fog = 1.0 - exp(-distance * frame.fog_color.a);
    if underwater {
        lit = lit * (1.0 - fog * 0.35);
    }
    color = color * (lit + direct * (1.0 - fresnel)) + specular;
    // Thin cutout leaves transmit light when the sun is behind them. The
    // view direction points toward the camera, so dot(view, sun) is negative
    // for a backlit leaf; attenuate by how directly the sunlight enters the
    // leaf and tint the transmission with the leaf texture.
    if is_foliage {
        let back_light = max(-dot(view_direction, sun_direction), 0.0);
        let leaf_facing = max(-dot(normal, sun_direction), 0.0);
        let transmission = pow(back_light, 3.0) * (0.30 + 0.70 * leaf_facing);
        // The visible side of a backlit leaf is in its own shadow, so step a
        // little way along the sun direction before sampling the shadow map.
        // This tests whether the far side of the leaf opens to direct sun.
        let transmission_visibility = sun_visibility(
            input.relative_position + sun_direction * 1.25,
            normal,
        );
        color = color
            + texel.rgb
                * frame.sun_color.rgb
                * (frame.sun_color.a * transmission_visibility * transmission * 1.15);
    }
    color = mix(color, frame.fog_color.rgb, fog);
    return vec4<f32>(color, alpha);
}
"#;

const HIGHLIGHT_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> @builtin(position) vec4<f32> {
    return camera.view_proj * vec4<f32>(input.position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.02, 0.02, 0.02, 0.9);
}
"#;

const COMPOSITE_SHADER: &str = r#"
@group(0) @binding(0)
var scene_texture: texture_2d<f32>;

@group(0) @binding(1)
var scene_sampler: sampler;

struct Output {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> Output {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let position = positions[index];
    var output: Output;
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.uv = vec2<f32>(
        position.x * 0.5 + 0.5,
        0.5 - position.y * 0.5,
    );
    return output;
}

@fragment
fn fs_main(input: Output) -> @location(0) vec4<f32> {
    return textureSampleLevel(scene_texture, scene_sampler, input.uv, 0.0);
}
"#;

const GODRAY_SHADER: &str = r#"
struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    sky_top: vec4<f32>,
    sky_horizon: vec4<f32>,
    fog_color: vec4<f32>,
    params: vec4<f32>,
    viewport: vec4<f32>,
    time: vec4<f32>,
    world_origin: vec4<f32>,
    camera_position: vec4<f32>,
    shadow_view_proj: mat4x4<f32>,
    probe_origin: vec4<f32>,
    probe_params: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> frame: Frame;

@group(1) @binding(0)
var scene_texture: texture_2d<f32>;

@group(1) @binding(1)
var scene_sampler: sampler;

struct GodrayOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) ray: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> GodrayOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let position = positions[index];
    var output: GodrayOutput;
    output.clip_position = vec4<f32>(position, 0.0, 1.0);
    output.uv = vec2<f32>(
        position.x * 0.5 + 0.5,
        0.5 - position.y * 0.5,
    );
    // The view-projection matrix is camera-relative, so the far point and
    // camera_position live in the same space.
    let far_point = frame.inv_view_proj * vec4<f32>(position, 1.0, 1.0);
    output.ray = normalize(far_point.xyz / max(far_point.w, 0.0001) - frame.camera_position.xyz);
    return output;
}

@fragment
fn fs_main(input: GodrayOutput) -> @location(0) vec4<f32> {
    let underwater = frame.time.y > 0.5;
    let sun_energy = max(frame.sun_color.a, 0.0);
    if (sun_energy < 0.02 || frame.params.x < 0.02) {
        return vec4<f32>(0.0);
    }

    let ray = normalize(input.ray);
    let sun_dir = normalize(frame.sun_direction.xyz);
    let to_sun = max(dot(ray, sun_dir), 0.0);
    // Above water the radial blur needs to be looking toward the sun. Under
    // water the shafts can cross the view even when the sun itself is above
    // the screen, which is the common first-person case.
    if (!underwater && to_sun < 0.015) {
        return vec4<f32>(0.0);
    }

    // Radial screen-space scattering. Above water this blurs bright sky
    // through leaf gaps; underwater the scene is already tinted, so a
    // synthetic caustic source keeps the light shafts readable.
    let sun_point = frame.camera_position.xyz + sun_dir * 256.0;
    let sun_clip = frame.view_proj * vec4<f32>(sun_point, 1.0);
    if (!underwater && sun_clip.w <= 0.001) {
        return vec4<f32>(0.0);
    }
    // Overhead sun is often exactly beside the view direction (w <= 0) for a
    // horizontal first-person camera. Underwater we fall back to a virtual
    // light center above the screen so the shafts still cross the view.
    var sun_uv = vec2<f32>(0.5, -0.45);
    if (sun_clip.w > 0.001) {
        let sun_ndc = sun_clip.xy / sun_clip.w;
        sun_uv = vec2<f32>(sun_ndc.x * 0.5 + 0.5, 0.5 - sun_ndc.y * 0.5);
    }
    let delta = sun_uv - input.uv;

    var rays = vec3<f32>(0.0);
    var total_weight = 0.0;
    let steps = 12u;
    for (var i: u32 = 0u; i < steps; i = i + 1u) {
        let t = (f32(i) + 0.5) / f32(steps);
        let sample_uv = clamp(input.uv + delta * t, vec2<f32>(0.001), vec2<f32>(0.999));
        let falloff = exp(-t * select(3.2, 2.1, underwater));
        if (underwater) {
            let dist_to_sun = length(sample_uv - sun_uv);
            let broad = exp(-dist_to_sun * 4.5);
            // Screen-space caustic bands keep the shafts readable when the
            // sun is off-screen; the broad term still brightens toward it.
            let stripe_a = 0.5 + 0.5 * sin(sample_uv.x * 23.0
                + sample_uv.y * 11.0 + frame.time.x * 0.42);
            let stripe_b = 0.5 + 0.5 * sin(sample_uv.y * 17.0
                - sample_uv.x * 7.0 - frame.time.x * 0.31);
            let source = 0.12 + 0.70 * stripe_a * stripe_b + broad * 0.55;
            rays = rays + vec3<f32>(0.34, 0.82, 0.92) * source * falloff;
        } else {
            let sample_color = textureSampleLevel(scene_texture, scene_sampler, sample_uv, 0.0).rgb;
            let luminance = dot(sample_color, vec3<f32>(0.2126, 0.7152, 0.0722));
            let source = smoothstep(0.38, 0.92, luminance);
            rays = rays + sample_color * source * falloff;
        }
        total_weight = total_weight + falloff;
    }
    rays = rays / max(total_weight, 0.001);
    let facing = select(pow(to_sun, 2.2), 0.28 + 0.72 * to_sun, underwater);
    if (underwater) {
        rays = rays * vec3<f32>(0.30, 0.78, 0.88) * (0.70 * sun_energy * facing);
    } else {
        rays = rays * frame.sun_color.rgb * (0.30 * sun_energy * facing);
    }
    return vec4<f32>(max(rays, vec3<f32>(0.0)), 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HighlightVertex {
    position: [f32; 3],
}

impl HighlightVertex {
    const fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
        }
    }
}

pub struct GpuState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub atlas: Arc<Atlas>,
    /// Adapter identity captured once so the F3 overlay can show which GPU and
    /// backend the frame is actually running on.
    pub adapter_info: wgpu::AdapterInfo,
    scene_texture: wgpu::Texture,
    scene_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    scene_bind_group: wgpu::BindGroup,
    opaque_scene_bind_group: wgpu::BindGroup,
    composite_pipeline: wgpu::RenderPipeline,
    godray_pipeline: wgpu::RenderPipeline,
    godray_frame_bind_group: wgpu::BindGroup,
    godray_scene_bind_group: wgpu::BindGroup,
    sky_pipeline: wgpu::RenderPipeline,
    voxel_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    highlight_pipeline: wgpu::RenderPipeline,
    voxel_bind_group: wgpu::BindGroup,
    probe_texture: wgpu::Texture,
    /// 探针网格最小角 (世界坐标) 与 UVW 缩放; w <= 0.5 表示未上传.
    probe_origin: [f32; 4],
    probe_params: [f32; 4],
    camera_buffer: wgpu::Buffer,
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_view: wgpu::TextureView,
    shadow_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    vertex_capacity: u64,
    index_capacity: u64,
    vertex_count: u32,
    index_count: u32,
    transparent_vertex_buffer: wgpu::Buffer,
    transparent_index_buffer: wgpu::Buffer,
    transparent_vertex_capacity: u64,
    transparent_index_capacity: u64,
    transparent_vertex_count: u32,
    transparent_index_count: u32,
    vertex_staging: Vec<GpuVertex>,
    opaque_batches: Vec<DrawBatch>,
    transparent_batches: Vec<DrawBatch>,
    visible_opaque: Vec<std::ops::Range<u32>>,
    visible_transparent: Vec<std::ops::Range<u32>>,
    shadow_opaque: Vec<std::ops::Range<u32>>,
    shadow_transparent: Vec<std::ops::Range<u32>>,
    last_mesh_upload_bytes: u64,
    highlight_buffer: wgpu::Buffer,
    highlight_capacity: u64,
    highlight_count: u32,
    start_time: Instant,
    /// `Surface<'static>` 要求窗口句柄一直有效，`Arc` 保活。
    #[allow(dead_code)]
    window: Arc<Window>,
}

/// Cheap snapshot of GPU-side mesh/resource usage for the F3 overlay.
///
/// This intentionally does not query the driver. It reports the buffers this
/// renderer owns plus texture sizes it knows it created, which is enough to
/// spot runaway LOD meshes or buffer growth at a glance.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuDebugStats {
    pub surface_width: u32,
    pub surface_height: u32,
    pub vertex_count: u32,
    pub index_count: u32,
    pub transparent_vertex_count: u32,
    pub transparent_index_count: u32,
    pub vertex_buffer_bytes: u64,
    pub index_buffer_bytes: u64,
    pub transparent_vertex_buffer_bytes: u64,
    pub transparent_index_buffer_bytes: u64,
    pub visible_indices: u32,
    pub shadow_indices: u32,
    pub scene_draw_calls: usize,
    pub shadow_draw_calls: usize,
    pub last_mesh_upload_bytes: u64,
    pub highlight_buffer_bytes: u64,
    pub scene_texture_bytes: u64,
    pub depth_texture_bytes: u64,
    pub shadow_texture_bytes: u64,
    pub atlas_texture_bytes: u64,
    pub probe_texture_bytes: u64,
}

impl GpuDebugStats {
    pub fn mesh_buffer_bytes(self) -> u64 {
        self.vertex_buffer_bytes
            + self.index_buffer_bytes
            + self.transparent_vertex_buffer_bytes
            + self.transparent_index_buffer_bytes
    }

    pub fn total_gpu_bytes(self) -> u64 {
        self.mesh_buffer_bytes()
            + self.highlight_buffer_bytes
            + self.scene_texture_bytes
            + self.depth_texture_bytes
            + self.shadow_texture_bytes
            + self.atlas_texture_bytes
            + self.probe_texture_bytes
    }

    pub fn opaque_triangles(self) -> u32 {
        self.index_count / 3
    }

    pub fn transparent_triangles(self) -> u32 {
        self.transparent_index_count / 3
    }
}

impl GpuState {
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size().max(PhysicalSize::new(1, 1));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("创建 Surface 失败");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("找不到可用 GPU adapter");
        let adapter_info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mc device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .expect("创建 Device 失败");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let atlas = Arc::new(
            atlas::build(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("assets/textures")
                    .as_path(),
            )
            .expect("构建体素纹理 atlas 失败"),
        );
        let (scene_texture, scene_view) = create_scene_texture(&device, size, config.format);
        let (probe_texture, probe_view, probe_sampler) = create_probe_volume(&device);
        let depth_view = create_depth_view(&device, size);
        let scene_bind_group_layout = create_scene_bind_group_layout(&device);
        let scene_bind_group =
            create_scene_bind_group(&device, &scene_bind_group_layout, &scene_view);
        let (_opaque_scene_texture, opaque_scene_view) =
            create_dummy_scene_texture(&device, &queue);
        let opaque_scene_bind_group =
            create_scene_bind_group(&device, &scene_bind_group_layout, &opaque_scene_view);
        let (
            sky_pipeline,
            voxel_pipeline,
            transparent_pipeline,
            highlight_pipeline,
            voxel_bind_group,
            camera_buffer,
            shadow_pipeline,
            shadow_view,
            shadow_bind_group,
        ) = create_voxel_pipeline(
            &device,
            &queue,
            &config,
            atlas.as_ref(),
            &scene_bind_group_layout,
            &ProbeBindings {
                view: &probe_view,
                sampler: &probe_sampler,
            },
        );
        let composite_pipeline =
            create_composite_pipeline(&device, &config, &scene_bind_group_layout);
        let godray_frame_bind_group_layout = create_godray_frame_bind_group_layout(&device);
        let godray_pipeline = create_godray_pipeline(
            &device,
            &config,
            &godray_frame_bind_group_layout,
            &scene_bind_group_layout,
        );
        let godray_frame_bind_group = create_godray_frame_bind_group(
            &device,
            &godray_frame_bind_group_layout,
            &camera_buffer,
        );
        let godray_scene_bind_group =
            create_scene_bind_group(&device, &scene_bind_group_layout, &scene_view);
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc voxel vertices"),
            size: std::mem::size_of::<GpuVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc voxel indices"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transparent_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc transparent voxel vertices"),
            size: std::mem::size_of::<GpuVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transparent_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc transparent voxel indices"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let highlight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc block highlight"),
            size: std::mem::size_of::<HighlightVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            surface,
            config,
            atlas,
            adapter_info,
            scene_texture,
            scene_view,
            depth_view,
            scene_bind_group_layout,
            scene_bind_group,
            opaque_scene_bind_group,
            composite_pipeline,
            godray_pipeline,
            godray_frame_bind_group,
            godray_scene_bind_group,
            sky_pipeline,
            voxel_pipeline,
            transparent_pipeline,
            highlight_pipeline,
            voxel_bind_group,
            probe_texture,
            probe_origin: [0.0; 4],
            probe_params: [0.0; 4],
            camera_buffer,
            shadow_pipeline,
            shadow_view,
            shadow_bind_group,
            vertex_buffer,
            index_buffer,
            vertex_count: 0,
            index_count: 0,
            vertex_capacity: std::mem::size_of::<GpuVertex>() as u64,
            index_capacity: std::mem::size_of::<u32>() as u64,
            vertex_staging: Vec::new(),
            opaque_batches: Vec::new(),
            transparent_batches: Vec::new(),
            visible_opaque: Vec::new(),
            visible_transparent: Vec::new(),
            shadow_opaque: Vec::new(),
            shadow_transparent: Vec::new(),
            last_mesh_upload_bytes: 0,
            highlight_buffer,
            highlight_capacity: std::mem::size_of::<HighlightVertex>() as u64,
            highlight_count: 0,
            transparent_vertex_buffer,
            transparent_index_buffer,
            transparent_vertex_count: 0,
            transparent_index_count: 0,
            transparent_vertex_capacity: std::mem::size_of::<GpuVertex>() as u64,
            transparent_index_capacity: std::mem::size_of::<u32>() as u64,
            start_time: Instant::now(),
            window,
        }
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        (self.scene_texture, self.scene_view) =
            create_scene_texture(&self.device, size, self.config.format);
        self.scene_bind_group = create_scene_bind_group(
            &self.device,
            &self.scene_bind_group_layout,
            &self.scene_view,
        );
        self.godray_scene_bind_group = create_scene_bind_group(
            &self.device,
            &self.scene_bind_group_layout,
            &self.scene_view,
        );
        self.depth_view = create_depth_view(&self.device, size);
    }

    /// Snapshot buffer sizes and current mesh counts without touching the GPU.
    pub fn debug_stats(&self) -> GpuDebugStats {
        let width = self.config.width as u64;
        let height = self.config.height as u64;
        let color_bytes = self.config.format.block_copy_size(None).unwrap_or(4) as u64;
        let atlas_side = self.atlas.size as u64;
        // Color + normal + roughness, each with a full mip chain (4/3).
        let atlas_texture_bytes = atlas_side * atlas_side * 4 * 3 * 4 / 3;
        GpuDebugStats {
            surface_width: self.config.width,
            surface_height: self.config.height,
            vertex_count: self.vertex_count,
            index_count: self.index_count,
            transparent_vertex_count: self.transparent_vertex_count,
            transparent_index_count: self.transparent_index_count,
            vertex_buffer_bytes: self.vertex_capacity,
            index_buffer_bytes: self.index_capacity,
            transparent_vertex_buffer_bytes: self.transparent_vertex_capacity,
            transparent_index_buffer_bytes: self.transparent_index_capacity,
            visible_indices: self
                .visible_opaque
                .iter()
                .chain(&self.visible_transparent)
                .map(|r| r.end - r.start)
                .sum(),
            shadow_indices: self
                .shadow_opaque
                .iter()
                .chain(&self.shadow_transparent)
                .map(|r| r.end - r.start)
                .sum(),
            scene_draw_calls: self.visible_opaque.len() + self.visible_transparent.len(),
            shadow_draw_calls: self.shadow_opaque.len() + self.shadow_transparent.len(),
            last_mesh_upload_bytes: self.last_mesh_upload_bytes,
            highlight_buffer_bytes: self.highlight_capacity,
            probe_texture_bytes: (probe::TEXELS as u64) * 8,
            scene_texture_bytes: width * height * color_bytes,
            depth_texture_bytes: width * height * 4,
            shadow_texture_bytes: (SHADOW_SIZE as u64) * (SHADOW_SIZE as u64) * 4,
            atlas_texture_bytes,
        }
    }

    pub fn upload_mesh(&mut self, mesh: &VoxelMesh) {
        self.opaque_batches = mesh_batches(&mesh.vertices, &mesh.indices);
        self.transparent_batches =
            mesh_batches(&mesh.transparent_vertices, &mesh.transparent_indices);
        self.last_mesh_upload_bytes = ((mesh.vertices.len() + mesh.transparent_vertices.len())
            * std::mem::size_of::<GpuVertex>()
            + (mesh.indices.len() + mesh.transparent_indices.len()) * std::mem::size_of::<u32>())
            as u64;
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            self.vertex_count = 0;
            self.index_count = 0;
        } else {
            self.vertex_count = mesh.vertices.len() as u32;
            self.vertex_staging.clear();
            self.vertex_staging
                .extend(mesh.vertices.iter().map(GpuVertex::from));
            let vertex_bytes = bytemuck::cast_slice(&self.vertex_staging);
            let index_bytes = bytemuck::cast_slice(&mesh.indices);
            if vertex_bytes.len() as u64 > self.vertex_capacity {
                self.vertex_capacity = vertex_bytes.len().next_power_of_two() as u64;
                self.vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mc voxel vertices"),
                    size: self.vertex_capacity,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            if index_bytes.len() as u64 > self.index_capacity {
                self.index_capacity = index_bytes.len().next_power_of_two() as u64;
                self.index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mc voxel indices"),
                    size: self.index_capacity,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            self.queue
                .write_buffer(&self.vertex_buffer, 0, vertex_bytes);
            self.queue.write_buffer(&self.index_buffer, 0, index_bytes);
            self.index_count = mesh.indices.len() as u32;
        }

        if mesh.transparent_vertices.is_empty() || mesh.transparent_indices.is_empty() {
            self.transparent_vertex_count = 0;
            self.transparent_index_count = 0;
        } else {
            self.transparent_vertex_count = mesh.transparent_vertices.len() as u32;
            self.vertex_staging.clear();
            self.vertex_staging
                .extend(mesh.transparent_vertices.iter().map(GpuVertex::from));
            let vertex_bytes = bytemuck::cast_slice(&self.vertex_staging);
            let index_bytes = bytemuck::cast_slice(&mesh.transparent_indices);
            if vertex_bytes.len() as u64 > self.transparent_vertex_capacity {
                self.transparent_vertex_capacity = vertex_bytes.len().next_power_of_two() as u64;
                self.transparent_vertex_buffer =
                    self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("mc transparent voxel vertices"),
                        size: self.transparent_vertex_capacity,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
            }
            if index_bytes.len() as u64 > self.transparent_index_capacity {
                self.transparent_index_capacity = index_bytes.len().next_power_of_two() as u64;
                self.transparent_index_buffer =
                    self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("mc transparent voxel indices"),
                        size: self.transparent_index_capacity,
                        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
            }
            self.queue
                .write_buffer(&self.transparent_vertex_buffer, 0, vertex_bytes);
            self.queue
                .write_buffer(&self.transparent_index_buffer, 0, index_bytes);
            self.transparent_index_count = mesh.transparent_indices.len() as u32;
        }
    }

    /// 把 CPU 烘焙的探针写入 3D 纹理; 原点按世界坐标记录, 采样时再转成
    /// camera-relative, 避免大坐标下的 f32 精度损失.
    pub fn upload_probes(&mut self, volume: &ProbeVolume) {
        let mut data = Vec::with_capacity(probe::TEXELS * 8);
        for texel in &volume.texels {
            for channel in texel {
                data.extend_from_slice(&half::f16::from_f32(*channel).to_le_bytes());
            }
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.probe_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((probe::DIM_X * 8) as u32),
                rows_per_image: Some(probe::DIM_Y as u32),
            },
            wgpu::Extent3d {
                width: probe::DIM_X as u32,
                height: probe::DIM_Y as u32,
                depth_or_array_layers: probe::DIM_Z as u32,
            },
        );
        self.probe_origin = [
            volume.origin.x as f32,
            volume.origin.y as f32,
            volume.origin.z as f32,
            0.0,
        ];
        let span = [
            (probe::SPACING * probe::DIM_X as i64) as f32,
            (probe::SPACING * probe::DIM_Y as i64) as f32,
            (probe::SPACING * probe::DIM_Z as i64) as f32,
        ];
        self.probe_params = [1.0 / span[0], 1.0 / span[1], 1.0 / span[2], 1.0];
    }

    pub fn update_camera(
        &mut self,
        view_proj: Mat4,
        underwater: bool,
        world_origin: glam::DVec3,
        camera_position: glam::DVec3,
    ) {
        // Photon 的视觉层次很大程度来自天空颜色、太阳色温和雾的联动。
        // 保持原版的 20 分钟完整昼夜循环（24,000 ticks）。
        let cycle = self.start_time.elapsed().as_secs_f32() * SUN_ANGULAR_SPEED;
        let sun_angle = cycle + std::f32::consts::FRAC_PI_2;
        let sun_direction = glam::Vec3::new(
            sun_angle.cos() * 0.78,
            sun_angle.sin(),
            sun_angle.cos() * 0.63,
        )
        .normalize();
        let daylight = ((sun_direction.y + 0.16) / 0.32).clamp(0.0, 1.0);
        let twilight = (1.0 - (sun_direction.y.abs() / 0.40).clamp(0.0, 1.0)).powi(2) * daylight;
        let sun_color = glam::Vec3::new(1.0, 0.96, 0.84)
            .lerp(glam::Vec3::new(1.0, 0.48, 0.18), twilight * 0.72);
        let night_top = glam::Vec3::new(0.002, 0.004, 0.012);
        let day_top = glam::Vec3::new(0.075, 0.24, 0.52);
        let night_horizon = glam::Vec3::new(0.009, 0.013, 0.028);
        let day_horizon = glam::Vec3::new(0.52, 0.66, 0.80);
        let sky_top = night_top.lerp(day_top, daylight);
        let sky_horizon = night_horizon.lerp(day_horizon, daylight);
        let fog_color = if underwater {
            glam::Vec3::new(0.035, 0.34, 0.40)
        } else {
            sky_horizon.lerp(sky_top, 0.22)
        };
        let shadow_view_proj = shadow_matrix(camera_position.as_vec3(), sun_direction);
        visible_ranges(
            &self.opaque_batches,
            view_proj,
            false,
            &mut self.visible_opaque,
        );
        visible_ranges(
            &self.transparent_batches,
            view_proj,
            false,
            &mut self.visible_transparent,
        );
        visible_ranges(
            &self.opaque_batches,
            shadow_view_proj,
            true,
            &mut self.shadow_opaque,
        );
        visible_ranges(
            &self.transparent_batches,
            shadow_view_proj,
            true,
            &mut self.shadow_transparent,
        );
        let inv_view_proj = view_proj.inverse();
        let uniform = FrameUniform {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            shadow_view_proj: shadow_view_proj.to_cols_array_2d(),
            sun_direction: [sun_direction.x, sun_direction.y, sun_direction.z, 1.0],
            sun_color: [sun_color.x, sun_color.y, sun_color.z, daylight * 0.92],
            sky_top: [sky_top.x, sky_top.y, sky_top.z, 0.18 + daylight * 0.30],
            sky_horizon: [
                sky_horizon.x,
                sky_horizon.y,
                sky_horizon.z,
                0.16 + daylight * 0.28,
            ],
            fog_color: [
                fog_color.x,
                fog_color.y,
                fog_color.z,
                if underwater {
                    0.018
                } else {
                    0.0018 + (1.0 - daylight) * 0.0022
                },
            ],
            params: [
                daylight,
                self.atlas.cols as f32,
                atlas::STRIDE as f32 / self.atlas.size as f32,
                atlas::GUTTER as f32 / self.atlas.size as f32,
            ],
            viewport: [
                self.config.width as f32,
                self.config.height as f32,
                0.0,
                0.0,
            ],
            time: [
                self.start_time.elapsed().as_secs_f32(),
                underwater as u8 as f32,
                0.0,
                0.0,
            ],
            world_origin: [
                world_origin.x as f32,
                world_origin.y as f32,
                world_origin.z as f32,
                0.0,
            ],
            camera_position: [
                camera_position.x as f32,
                camera_position.y as f32,
                camera_position.z as f32,
                0.0,
            ],
            probe_origin: [
                self.probe_origin[0] - world_origin.x as f32,
                self.probe_origin[1] - world_origin.y as f32,
                self.probe_origin[2] - world_origin.z as f32,
                0.0,
            ],
            probe_params: self.probe_params,
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    pub fn draw_voxels<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.index_count == 0 {
            return;
        }
        pass.set_pipeline(&self.voxel_pipeline);
        pass.set_bind_group(2, &self.shadow_bind_group, &[]);
        pass.set_bind_group(0, &self.voxel_bind_group, &[]);
        pass.set_bind_group(1, &self.opaque_scene_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        for range in &self.visible_opaque {
            pass.draw_indexed(range.clone(), 0, 0..1);
        }
    }

    pub fn draw_sky<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.sky_pipeline);
        pass.set_bind_group(2, &self.shadow_bind_group, &[]);
        pass.set_bind_group(0, &self.voxel_bind_group, &[]);
        // The shared pipeline layout declares the scene group at index 1.
        // The sky shader does not sample it, but wgpu still requires every
        // declared bind group to be set before issuing the draw.
        pass.set_bind_group(1, &self.opaque_scene_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    pub fn draw_transparent_voxels<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.transparent_index_count == 0 {
            return;
        }
        pass.set_pipeline(&self.transparent_pipeline);
        pass.set_bind_group(2, &self.shadow_bind_group, &[]);
        pass.set_bind_group(0, &self.voxel_bind_group, &[]);
        pass.set_bind_group(1, &self.scene_bind_group, &[]);
        pass.set_vertex_buffer(0, self.transparent_vertex_buffer.slice(..));
        pass.set_index_buffer(
            self.transparent_index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        for range in &self.visible_transparent {
            pass.draw_indexed(range.clone(), 0, 0..1);
        }
    }

    pub fn draw_scene<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, &self.scene_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    pub fn draw_god_rays<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.godray_pipeline);
        pass.set_bind_group(0, &self.godray_frame_bind_group, &[]);
        pass.set_bind_group(1, &self.godray_scene_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    pub fn scene_view(&self) -> &wgpu::TextureView {
        &self.scene_view
    }

    pub fn draw_shadows(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mc sun shadows"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.shadow_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(0, &self.voxel_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        for range in &self.shadow_opaque {
            pass.draw_indexed(range.clone(), 0, 0..1);
        }
        // Glass detail casts a cutout shadow; water does not cast an opaque one.
        pass.set_vertex_buffer(0, self.transparent_vertex_buffer.slice(..));
        pass.set_index_buffer(
            self.transparent_index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        for range in &self.shadow_transparent {
            pass.draw_indexed(range.clone(), 0, 0..1);
        }
    }

    pub fn upload_highlight(&mut self, block: Option<(i64, i64, i64)>, origin: glam::DVec3) {
        let Some((x, y, z)) = block else {
            self.highlight_count = 0;
            return;
        };
        let min =
            glam::DVec3::new(x as f64, y as f64, z as f64) - origin - glam::DVec3::splat(0.002);
        let max = min + glam::DVec3::splat(1.004);
        let corners = [
            [min.x as f32, min.y as f32, min.z as f32],
            [max.x as f32, min.y as f32, min.z as f32],
            [max.x as f32, min.y as f32, max.z as f32],
            [min.x as f32, min.y as f32, max.z as f32],
            [min.x as f32, max.y as f32, min.z as f32],
            [max.x as f32, max.y as f32, min.z as f32],
            [max.x as f32, max.y as f32, max.z as f32],
            [min.x as f32, max.y as f32, max.z as f32],
        ];
        let edge_indices = [
            0, 1, 1, 2, 2, 3, 3, 0, // bottom
            4, 5, 5, 6, 6, 7, 7, 4, // top
            0, 4, 1, 5, 2, 6, 3, 7, // sides
        ];
        let vertices: Vec<_> = edge_indices
            .into_iter()
            .map(|index| HighlightVertex {
                position: corners[index],
            })
            .collect();
        let bytes = bytemuck::cast_slice(&vertices);
        if bytes.len() as u64 > self.highlight_capacity {
            self.highlight_capacity = bytes.len().next_power_of_two() as u64;
            self.highlight_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc block highlight"),
                size: self.highlight_capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        self.queue.write_buffer(&self.highlight_buffer, 0, bytes);
        self.highlight_count = vertices.len() as u32;
    }

    pub fn draw_highlight<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.highlight_count == 0 {
            return;
        }
        pass.set_pipeline(&self.highlight_pipeline);
        pass.set_bind_group(2, &self.shadow_bind_group, &[]);
        pass.set_bind_group(0, &self.voxel_bind_group, &[]);
        pass.set_bind_group(1, &self.opaque_scene_bind_group, &[]);
        pass.set_vertex_buffer(0, self.highlight_buffer.slice(..));
        pass.draw(0..self.highlight_count, 0..1);
    }

    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// 保留给无世界状态的简单清屏路径。
    pub fn render(&mut self) -> bool {
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return true;
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                let size = PhysicalSize::new(self.config.width, self.config.height);
                self.resize(size);
                return true;
            }
            wgpu::CurrentSurfaceTexture::Validation => return true,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mc clear frame"),
            });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mc clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(MENU_CLEAR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(output);
        true
    }
}

fn create_scene_texture(
    device: &wgpu::Device,
    size: PhysicalSize<u32>,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mc scene color"),
        size: wgpu::Extent3d {
            width: size.width.max(1),
            height: size.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_scene_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mc scene bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn create_scene_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("mc scene sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mc scene bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    })
}

fn create_dummy_scene_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mc opaque dummy scene"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &[0, 0, 0, 255],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// 创建探针 3D 纹理 (Rgba16Float SH L1) 与线性采样器.
fn create_probe_volume(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mc irradiance probes"),
        size: wgpu::Extent3d {
            width: probe::DIM_X as u32,
            height: probe::DIM_Y as u32,
            depth_or_array_layers: probe::DIM_Z as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("mc probe sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    (texture, view, sampler)
}

fn create_depth_view(device: &wgpu::Device, size: PhysicalSize<u32>) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("mc depth"),
            size: wgpu::Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn upload_atlas_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    source: &image::RgbaImage,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let mip_level_count = source.width().max(source.height()).ilog2() + 1;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: source.width(),
            height: source.height(),
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let mut mip = source.clone();
    for level in 0..mip_level_count {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            mip.as_raw(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * mip.width()),
                rows_per_image: Some(mip.height()),
            },
            wgpu::Extent3d {
                width: mip.width(),
                height: mip.height(),
                depth_or_array_layers: 1,
            },
        );
        if level + 1 < mip_level_count {
            mip = image::imageops::resize(
                &mip,
                (mip.width() / 2).max(1),
                (mip.height() / 2).max(1),
                image::imageops::FilterType::Lanczos3,
            );
        }
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Orthographic light camera, snapped to shadow texels to limit crawling.
/// All inputs stay relative to the mesh origin, including at large world coordinates.
fn shadow_matrix(camera: glam::Vec3, sun: glam::Vec3) -> Mat4 {
    // Z stays away from the entire daily sun trajectory, so the basis does
    // not flip near noon. The fallback also supports arbitrary test lights.
    let up = if sun.z.abs() < 0.95 {
        glam::Vec3::Z
    } else {
        glam::Vec3::Y
    };
    let rotation = Mat4::look_to_rh(glam::Vec3::ZERO, -sun, up);
    let mut center = rotation.transform_point3(camera);
    let texel = SHADOW_RADIUS * 2.0 / SHADOW_SIZE as f32;
    center.x = (center.x / texel).round() * texel;
    center.y = (center.y / texel).round() * texel;
    let target = rotation.inverse().transform_point3(center);
    let view = Mat4::look_at_rh(target + sun * 384.0, target, up);
    Mat4::orthographic_rh(
        -SHADOW_RADIUS,
        SHADOW_RADIUS,
        -SHADOW_RADIUS,
        SHADOW_RADIUS,
        0.1,
        768.0,
    ) * view
}

#[cfg(test)]
#[path = "lighting_tests.rs"]
mod lighting_tests;

#[allow(clippy::type_complexity)]
/// 探针 3D 纹理与采样器的绑定, 分组传参避免函数签名过长.
struct ProbeBindings<'a> {
    view: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
}

fn create_voxel_pipeline(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &wgpu::SurfaceConfiguration,
    atlas: &Atlas,
    scene_bind_group_layout: &wgpu::BindGroupLayout,
    probe: &ProbeBindings<'_>,
) -> (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::BindGroup,
    wgpu::Buffer,
    wgpu::RenderPipeline,
    wgpu::TextureView,
    wgpu::BindGroup,
) {
    create_voxel_pipeline_with_shader(
        device,
        queue,
        config,
        atlas,
        scene_bind_group_layout,
        probe,
        VOXEL_SHADER,
    )
}

fn create_voxel_pipeline_with_shader(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: &wgpu::SurfaceConfiguration,
    atlas: &Atlas,
    scene_bind_group_layout: &wgpu::BindGroupLayout,
    probe: &ProbeBindings<'_>,
    source: &str,
) -> (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::BindGroup,
    wgpu::Buffer,
    wgpu::RenderPipeline,
    wgpu::TextureView,
    wgpu::BindGroup,
) {
    let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mc analytic sky shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SKY_SHADER)),
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mc voxel shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mc camera uniform"),
        size: std::mem::size_of::<FrameUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let (_texture, texture_view) = upload_atlas_texture(
        device,
        queue,
        "mc block albedo atlas",
        &atlas.image,
        wgpu::TextureFormat::Rgba8UnormSrgb,
    );
    let (_normal_texture, normal_view) = upload_atlas_texture(
        device,
        queue,
        "mc block normal atlas",
        &atlas.normal_image,
        wgpu::TextureFormat::Rgba8Unorm,
    );
    let (_roughness_texture, roughness_view) = upload_atlas_texture(
        device,
        queue,
        "mc block roughness atlas",
        &atlas.roughness_image,
        wgpu::TextureFormat::Rgba8Unorm,
    );
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("mc block sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        anisotropy_clamp: 8,
        ..Default::default()
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mc voxel bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mc voxel bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&normal_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&roughness_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(probe.view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(probe.sampler),
            },
        ],
    });
    let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mc sun shadow map"),
        size: wgpu::Extent3d {
            width: SHADOW_SIZE,
            height: SHADOW_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let shadow_view = shadow_texture.create_view(&Default::default());
    let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("mc shadow comparison sampler"),
        compare: Some(wgpu::CompareFunction::LessEqual),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let shadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mc shadow sampling layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
        ],
    });
    let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mc shadow sampling"),
        layout: &shadow_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&shadow_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&shadow_sampler),
            },
        ],
    });
    let shadow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mc shadow render layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mc alpha-tested sun shadows"),
        layout: Some(&shadow_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_shadow"),
            compilation_options: Default::default(),
            buffers: &[Some(GpuVertex::desc())],
        },
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: wgpu::DepthBiasState {
                constant: 2,
                slope_scale: 1.5,
                clamp: 0.0,
            },
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_shadow"),
            compilation_options: Default::default(),
            targets: &[],
        }),
        multiview_mask: None,
        cache: None,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mc voxel pipeline layout"),
        bind_group_layouts: &[
            Some(&bind_group_layout),
            Some(scene_bind_group_layout),
            Some(&shadow_layout),
        ],
        immediate_size: 0,
    });
    let create_voxel_render_pipeline =
        |layout: &wgpu::PipelineLayout,
         label: &'static str,
         depth_write_enabled: bool,
         blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(GpuVertex::desc())],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(depth_write_enabled),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
    let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mc analytic sky pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &sky_shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &sky_shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let pipeline = create_voxel_render_pipeline(&pipeline_layout, "mc voxel pipeline", true, None);
    let transparent_pipeline = create_voxel_render_pipeline(
        &pipeline_layout,
        "mc transparent voxel pipeline",
        false,
        Some(wgpu::BlendState::ALPHA_BLENDING),
    );

    let highlight_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mc block highlight shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(HIGHLIGHT_SHADER)),
    });
    let highlight_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mc block highlight pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &highlight_shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(HighlightVertex::desc())],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            // wgpu 不允许 LineList 使用三角形专用的 depth bias。
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &highlight_shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    (
        sky_pipeline,
        pipeline,
        transparent_pipeline,
        highlight_pipeline,
        bind_group,
        camera_buffer,
        shadow_pipeline,
        shadow_view,
        shadow_bind_group,
    )
}

fn create_composite_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    scene_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mc scene composite shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COMPOSITE_SHADER)),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mc scene composite pipeline layout"),
        bind_group_layouts: &[Some(scene_bind_group_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mc scene composite pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_godray_frame_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mc godray frame bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn create_godray_frame_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mc godray frame bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    })
}

fn create_godray_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    frame_bind_group_layout: &wgpu::BindGroupLayout,
    scene_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mc godray shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GODRAY_SHADER)),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mc godray pipeline layout"),
        bind_group_layouts: &[Some(frame_bind_group_layout), Some(scene_bind_group_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mc godray pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}
