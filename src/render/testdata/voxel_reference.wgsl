// Pre-optimization sampling reference; indirect lighting tracks the production model.

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
    var visibility = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            visibility += textureSampleCompareLevel(shadow_map, shadow_sampler,
                uv + vec2<f32>(f32(x), f32(y)) * texel, clip.z - 0.00004);
        }
    }
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
    let is_water = input.material > 0.5;
    let world_xz = input.position.xz + frame.world_origin.xz;
    let top_wave = water_height(world_xz, frame.time.x);
    if is_water && input.normal.y > 0.5 {
        position.y = position.y + top_wave;
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
    if textureSample(atlas_texture, atlas_sampler, input.uv).a < 0.5 || input.material > 0.5 {
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
    let h = 0.08;
    let p = world_position.xz;
    let dx = water_height(p + vec2<f32>(h, 0.0), time)
        - water_height(p - vec2<f32>(h, 0.0), time);
    let dz = water_height(p + vec2<f32>(0.0, h), time)
        - water_height(p - vec2<f32>(0.0, h), time);
    return normalize(vec3<f32>(-dx, 2.0 * h, -dz));
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
    weights *= vec4<f32>(
        select(0.0, 1.0, input.blend_tiles.x != current_tile),
        select(0.0, 1.0, input.blend_tiles.y != current_tile),
        select(0.0, 1.0, input.blend_tiles.z != current_tile),
        select(0.0, 1.0, input.blend_tiles.w != current_tile)
    );
    let edge_weight = dot(weights, vec4<f32>(1.0));
    if edge_weight > 0.0001 {
        let uv0 = packed_atlas_uv(input.blend_tiles.x, input.local_uv, input.texture_region);
        let uv1 = packed_atlas_uv(input.blend_tiles.y, input.local_uv, input.texture_region);
        let uv2 = packed_atlas_uv(input.blend_tiles.z, input.local_uv, input.texture_region);
        let uv3 = packed_atlas_uv(input.blend_tiles.w, input.local_uv, input.texture_region);
        let denominator = 1.0 + edge_weight;
        texel = (
            texel
            + textureSample(atlas_texture, atlas_sampler, uv0) * weights.x
            + textureSample(atlas_texture, atlas_sampler, uv1) * weights.y
            + textureSample(atlas_texture, atlas_sampler, uv2) * weights.z
            + textureSample(atlas_texture, atlas_sampler, uv3) * weights.w
        ) / denominator;
        normal_texel = (
            normal_texel
            + textureSample(atlas_normal, atlas_sampler, uv0) * weights.x
            + textureSample(atlas_normal, atlas_sampler, uv1) * weights.y
            + textureSample(atlas_normal, atlas_sampler, uv2) * weights.z
            + textureSample(atlas_normal, atlas_sampler, uv3) * weights.w
        ) / denominator;
        roughness = (
            roughness
            + textureSample(atlas_roughness, atlas_sampler, uv0).r * weights.x
            + textureSample(atlas_roughness, atlas_sampler, uv1).r * weights.y
            + textureSample(atlas_roughness, atlas_sampler, uv2).r * weights.z
            + textureSample(atlas_roughness, atlas_sampler, uv3).r * weights.w
        ) / denominator;
    }
    // Transparent texture pixels must not write depth. Glass is represented by
    // an opaque detail pattern on a transparent background; without this test,
    // the alpha-blended color is invisible but the depth value still hides the
    // terrain behind it, producing large clear-colored planes.
    if texel.a < 0.5 {
        discard;
    }
    let is_water = input.material > 0.5;
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
    let tangent_normal = normal_texel.xyz * 2.0 - vec3<f32>(1.0);
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
    color = mix(color, frame.fog_color.rgb, fog);
    return vec4<f32>(color, alpha);
}
