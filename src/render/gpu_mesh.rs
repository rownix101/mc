//! GPU vertex encoding and conservative visibility of contiguous mesh batches.

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

use super::voxel::Vertex;

/// Axis-aligned voxel frames encode exactly as signed normalized bytes.
/// Positions, UVs, AO and material IDs retain their original precision.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GpuVertex {
    position: [f32; 3],
    uv: [f32; 2],
    normal: [i8; 4],
    ao: f32,
    material: f32,
    tangent: [i8; 4],
    bitangent: [i8; 4],
    local_uv: [f32; 2],
    blend_tiles: [u32; 4],
    texture_region: [u32; 4],
    light: f32,
}

impl From<&Vertex> for GpuVertex {
    fn from(vertex: &Vertex) -> Self {
        let encode = |v: [f32; 3]| {
            let [x, y, z] = v.map(|v| (v.clamp(-1.0, 1.0) * 127.0).round() as i8);
            [x, y, z, 0]
        };
        Self {
            position: vertex.position,
            uv: vertex.uv,
            normal: encode(vertex.normal),
            ao: vertex.ao,
            light: vertex.light,
            material: vertex.material,
            tangent: encode(vertex.tangent),
            bitangent: encode(vertex.bitangent),
            local_uv: vertex.local_uv,
            blend_tiles: vertex.blend_tiles,
            texture_region: vertex.texture_region,
        }
    }
}

impl GpuVertex {
    pub(super) const fn desc() -> wgpu::VertexBufferLayout<'static> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 11] = wgpu::vertex_attr_array![
            0 => Float32x3, 1 => Float32x2, 2 => Snorm8x4,
            3 => Float32, 4 => Float32, 5 => Snorm8x4, 6 => Snorm8x4,
            7 => Float32x2, 8 => Uint32x4, 9 => Uint32x4, 10 => Float32
        ];
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

/// Batches follow mesh order; no triangle reordering is needed for blending.
pub(super) struct DrawBatch {
    indices: Range<u32>,
    min: Vec3,
    max: Vec3,
    casts_shadow: bool,
}

pub(super) fn mesh_batches(vertices: &[Vertex], indices: &[u32]) -> Vec<DrawBatch> {
    const INDICES_PER_BATCH: usize = 256 * 6;
    indices
        .chunks(INDICES_PER_BATCH)
        .enumerate()
        .map(|(batch, indices)| {
            let mut min = Vec3::splat(f32::INFINITY);
            let mut max = Vec3::splat(f32::NEG_INFINITY);
            let mut water = false;
            let mut casts_shadow = false;
            for &index in indices {
                let vertex = &vertices[index as usize];
                let position = Vec3::from_array(vertex.position);
                min = min.min(position);
                max = max.max(position);
                // Material 1 is water. Leaves use 2 because they still cast
                // alpha-tested shadows, while water never enters the shadow
                // pass.
                water |= (vertex.material - 1.0).abs() < 0.5;
                casts_shadow |= (vertex.material - 1.0).abs() >= 0.5;
            }
            // water_height in the vertex shader has total amplitude 0.036.
            // Keep a little slack for floating-point error at frustum edges.
            let padding = Vec3::new(0.001, if water { 0.04 } else { 0.001 }, 0.001);
            let start = (batch * INDICES_PER_BATCH) as u32;
            DrawBatch {
                indices: start..start + indices.len() as u32,
                min: min - padding,
                max: max + padding,
                casts_shadow,
            }
        })
        .collect()
}

struct Frustum([Vec4; 6]);

impl Frustum {
    fn new(matrix: Mat4) -> Self {
        let rows = matrix.transpose();
        // WGPU clip depth is 0 <= z <= w (not OpenGL's -w <= z <= w).
        Self([
            rows.w_axis + rows.x_axis,
            rows.w_axis - rows.x_axis,
            rows.w_axis + rows.y_axis,
            rows.w_axis - rows.y_axis,
            rows.z_axis,
            rows.w_axis - rows.z_axis,
        ])
    }

    fn intersects(&self, min: Vec3, max: Vec3) -> bool {
        let center = (min + max) * 0.5;
        let extent = (max - min) * 0.5;
        self.0.iter().all(|plane| {
            plane.truncate().dot(center) + plane.w + plane.truncate().abs().dot(extent) >= 0.0
        })
    }
}

pub(super) fn visible_ranges(
    batches: &[DrawBatch],
    matrix: Mat4,
    shadows: bool,
    result: &mut Vec<Range<u32>>,
) {
    result.clear();
    let frustum = Frustum::new(matrix);
    for batch in batches {
        if (shadows && !batch.casts_shadow) || !frustum.intersects(batch.min, batch.max) {
            continue;
        }
        if let Some(previous) = result.last_mut()
            && previous.end == batch.indices.start
        {
            previous.end = batch.indices.end;
        } else {
            result.push(batch.indices.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_vertex_preserves_voxel_frames() {
        assert_eq!(std::mem::size_of::<Vertex>(), 108);
        assert_eq!(std::mem::size_of::<GpuVertex>(), 84);
        for axis in [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, -Vec3::Y, -Vec3::Z] {
            let vertex = Vertex {
                normal: axis.to_array(),
                tangent: (-axis).to_array(),
                bitangent: axis.to_array(),
                ..Vertex::zeroed()
            };
            let packed = GpuVertex::from(&vertex);
            for i in 0..3 {
                assert_eq!(packed.normal[i] as f32 / 127.0, vertex.normal[i]);
                assert_eq!(packed.tangent[i] as f32 / 127.0, vertex.tangent[i]);
                assert_eq!(packed.bitangent[i] as f32 / 127.0, vertex.bitangent[i]);
            }
        }
    }

    #[test]
    fn frustum_keeps_intersections_and_rejects_all_six_outside_planes() {
        let frustum = Frustum::new(Mat4::IDENTITY);
        assert!(frustum.intersects(Vec3::splat(-2.0), Vec3::splat(2.0)));
        assert!(frustum.intersects(Vec3::new(-0.1, -0.1, -0.1), Vec3::splat(0.1)));
        for point in [
            Vec3::X * 2.0,
            -Vec3::X * 2.0,
            Vec3::Y * 2.0,
            -Vec3::Y * 2.0,
            Vec3::Z * 2.0,
            -Vec3::Z,
        ] {
            assert!(!frustum.intersects(point - Vec3::splat(0.1), point + Vec3::splat(0.1)));
        }
        let perspective = Frustum::new(Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0));
        assert!(perspective.intersects(Vec3::new(-1.0, -1.0, -5.0), Vec3::new(1.0, 1.0, -3.0)));
        assert!(!perspective.intersects(Vec3::splat(1.0), Vec3::splat(2.0)));
    }

    #[test]
    fn draw_ranges_preserve_order_merge_neighbors_and_exclude_water_shadows() {
        let batches: Vec<_> = (0..4)
            .map(|i| DrawBatch {
                indices: i * 6..(i + 1) * 6,
                min: Vec3::new(-0.1, -0.1, 0.1),
                max: Vec3::new(0.1, 0.1, 0.9),
                casts_shadow: i != 2,
            })
            .collect();
        let mut ranges = Vec::new();
        visible_ranges(&batches, Mat4::IDENTITY, false, &mut ranges);
        assert_eq!(ranges, vec![0..24]);
        visible_ranges(&batches, Mat4::IDENTITY, true, &mut ranges);
        assert_eq!(ranges, vec![0..12, 18..24]);
        visible_ranges(&[], Mat4::IDENTITY, false, &mut ranges);
        assert!(ranges.is_empty());
    }

    #[test]
    fn water_bounds_include_shader_displacement() {
        let vertex = Vertex {
            position: [0.0, 1.02, 0.5],
            material: 1.0,
            ..Vertex::zeroed()
        };
        let batches = mesh_batches(&[vertex], &[0, 0, 0]);
        let mut ranges = Vec::new();
        visible_ranges(&batches, Mat4::IDENTITY, false, &mut ranges);
        assert_eq!(ranges, vec![0..3]);
        visible_ranges(&batches, Mat4::IDENTITY, true, &mut ranges);
        assert!(ranges.is_empty());
    }

    #[test]
    fn cutout_leaves_are_not_water_and_keep_casting_shadows() {
        let vertex = Vertex {
            position: [0.0, 0.5, 0.5],
            material: 2.0,
            ..Vertex::zeroed()
        };
        let batches = mesh_batches(&[vertex], &[0, 0, 0]);
        let mut ranges = Vec::new();
        visible_ranges(&batches, Mat4::IDENTITY, false, &mut ranges);
        assert_eq!(ranges, vec![0..3]);
        // Leaves must submit to the shadow pass; only the shader excludes
        // water. Their atlas alpha test then creates dappled shadows.
        visible_ranges(&batches, Mat4::IDENTITY, true, &mut ranges);
        assert_eq!(ranges, vec![0..3]);
    }
}
