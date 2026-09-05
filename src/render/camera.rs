//! 第一人称透视相机。

use glam::{DVec3, Mat4, Vec3};

use crate::player::physics::Player;

pub struct Camera {
    pub yaw: f64,
    pub pitch: f64,
    pub sensitivity: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            // yaw=0 朝向 -Z，符合体素世界的常用初始方向。
            yaw: 0.0,
            pitch: 0.0,
            sensitivity: 0.0025,
        }
    }
}

impl Camera {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn mouse_motion(&mut self, dx: f64, dy: f64) {
        self.yaw -= dx * self.sensitivity;
        self.pitch = (self.pitch - dy * self.sensitivity).clamp(-1.54, 1.54);
    }

    pub fn forward(&self) -> DVec3 {
        let cos_pitch = self.pitch.cos();
        DVec3::new(
            self.yaw.sin() * cos_pitch,
            self.pitch.sin(),
            -self.yaw.cos() * cos_pitch,
        )
    }

    pub fn eye_position(&self, player: &Player) -> DVec3 {
        player.position + DVec3::new(0.0, 1.62, 0.0)
    }

    pub fn view_projection(
        &self,
        player: &Player,
        mesh_origin: DVec3,
        width: u32,
        height: u32,
    ) -> Mat4 {
        let eye = self.eye_position(player) - mesh_origin;
        let target = (eye + self.forward()).as_vec3();
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        let view = Mat4::look_at_rh(eye.as_vec3(), target, Vec3::Y);
        let projection = Mat4::perspective_rh_gl(75.0_f32.to_radians(), aspect, 0.05, 256.0);
        projection * view
    }
}
