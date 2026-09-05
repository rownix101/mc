//! 最小体素玩家物理。
//!
//! 坐标约定：玩家位置是脚底中心，Y 轴向上；方块占据
//! `[x, x + 1) × [y, y + 1) × [z, z + 1)`。碰撞采用轴分离的 AABB
//! 求解，足够支持体素世界中的行走、重力、跳跃和贴墙滑动。

use glam::DVec3;

use crate::world::voxel::VoxelWorld;

const COLLISION_EPSILON: f64 = 1.0e-7;

/// 玩家移动与跳跃输入。数值通常在 `[-1, 1]`，物理层会再次归一化。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlayerInput {
    /// 左右，正数向 +X。
    pub move_x: f64,
    /// 前后，正数向 +Z；当前 UI 的 W 会传入负数。
    pub move_z: f64,
    /// 跳跃请求，只在一个固定物理步中消费。
    pub jump: bool,
    /// 在流体中向上游泳。通常由持续按住空格产生。
    pub swim_up: bool,
    /// 在流体中向下游泳。通常由持续按住 Shift 产生。
    pub swim_down: bool,
}

/// 体素玩家的 AABB。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: DVec3,
    pub max: DVec3,
}

impl Aabb {
    pub fn new(min: DVec3, max: DVec3) -> Self {
        Self { min, max }
    }

    fn overlaps_block(&self, x: i64, y: i64, z: i64) -> bool {
        let min = DVec3::new(x as f64, y as f64, z as f64);
        let max = min + DVec3::ONE;
        self.max.x > min.x + COLLISION_EPSILON
            && self.min.x < max.x - COLLISION_EPSILON
            && self.max.y > min.y + COLLISION_EPSILON
            && self.min.y < max.y - COLLISION_EPSILON
            && self.max.z > min.z + COLLISION_EPSILON
            && self.min.z < max.z - COLLISION_EPSILON
    }
}

/// 一次物理步的结果，便于相机、音效和调试 UI 使用。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepResult {
    pub collided_x: bool,
    pub collided_y: bool,
    pub collided_z: bool,
    pub landed: bool,
    pub hit_ceiling: bool,
}

/// 玩家物理参数，单位是“方块 / 秒”。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicsConfig {
    pub width: f64,
    pub height: f64,
    pub walk_speed: f64,
    pub ground_acceleration: f64,
    pub air_acceleration: f64,
    pub ground_friction: f64,
    pub air_friction: f64,
    pub gravity: f64,
    pub jump_speed: f64,
    pub terminal_velocity: f64,
    pub swim_speed: f64,
    pub swim_acceleration: f64,
    pub swim_friction: f64,
    pub swim_vertical_speed: f64,
    pub swim_vertical_acceleration: f64,
    pub swim_vertical_friction: f64,
    pub water_gravity: f64,
    pub water_terminal_velocity: f64,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            width: 0.6,
            height: 1.8,
            walk_speed: 4.3,
            ground_acceleration: 35.0,
            air_acceleration: 12.0,
            ground_friction: 18.0,
            air_friction: 1.5,
            gravity: -32.0,
            jump_speed: 8.5,
            terminal_velocity: -78.0,
            swim_speed: 3.0,
            swim_acceleration: 10.0,
            swim_friction: 5.0,
            swim_vertical_speed: 3.5,
            swim_vertical_acceleration: 12.0,
            swim_vertical_friction: 8.0,
            water_gravity: -8.0,
            water_terminal_velocity: -4.0,
        }
    }
}

/// 可受重力、移动和体素碰撞影响的玩家。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Player {
    /// 脚底中心位置，不是 AABB 中心。
    pub position: DVec3,
    pub velocity: DVec3,
    pub on_ground: bool,
    pub config: PhysicsConfig,
}

impl Player {
    pub fn new(position: DVec3) -> Self {
        Self::with_config(position, PhysicsConfig::default())
    }

    pub fn with_config(position: DVec3, config: PhysicsConfig) -> Self {
        Self {
            position,
            velocity: DVec3::ZERO,
            on_ground: false,
            config,
        }
    }

    pub fn aabb(&self) -> Aabb {
        let half_width = self.config.width * 0.5;
        Aabb::new(
            DVec3::new(
                self.position.x - half_width,
                self.position.y,
                self.position.z - half_width,
            ),
            DVec3::new(
                self.position.x + half_width,
                self.position.y + self.config.height,
                self.position.z + half_width,
            ),
        )
    }

    /// 推进一个固定物理步。
    pub fn step<W: VoxelWorld>(
        &mut self,
        world: &mut W,
        input: PlayerInput,
        dt: f64,
    ) -> StepResult {
        let dt = dt.clamp(0.0, 0.1);
        if dt == 0.0 {
            return StepResult::default();
        }

        let mut result = StepResult::default();
        let was_on_ground = self.on_ground;
        let in_fluid = intersects_fluid(world, self.aabb());
        let submerged = in_fluid && head_intersects_fluid(world, self.aabb());

        let mut direction = DVec3::new(input.move_x, 0.0, input.move_z);
        if direction.length_squared() > 1.0 {
            direction = direction.normalize();
        }
        let (speed, acceleration, friction) = if in_fluid {
            (
                self.config.swim_speed,
                self.config.swim_acceleration,
                self.config.swim_friction,
            )
        } else if was_on_ground {
            (
                self.config.walk_speed,
                self.config.ground_acceleration,
                self.config.ground_friction,
            )
        } else {
            (
                self.config.walk_speed,
                self.config.air_acceleration,
                self.config.air_friction,
            )
        };
        let target_velocity = direction * speed;

        if direction.length_squared() > 0.0 {
            self.velocity.x = approach(self.velocity.x, target_velocity.x, acceleration * dt);
            self.velocity.z = approach(self.velocity.z, target_velocity.z, acceleration * dt);
        } else {
            self.velocity.x = approach(self.velocity.x, 0.0, friction * dt);
            self.velocity.z = approach(self.velocity.z, 0.0, friction * dt);
        }

        if submerged {
            // `jump` is also accepted as a one-step upward impulse for callers
            // that do not track held-key state separately.
            let swim_vertical_input = (input.swim_up || input.jump) as i8 - input.swim_down as i8;
            let target_vertical_velocity =
                swim_vertical_input as f64 * self.config.swim_vertical_speed;
            let vertical_change = if swim_vertical_input == 0 {
                self.config.swim_vertical_friction
            } else {
                self.config.swim_vertical_acceleration
            };
            self.velocity.y = approach(
                self.velocity.y,
                target_vertical_velocity,
                vertical_change * dt,
            );
            self.on_ground = false;
        } else if in_fluid {
            // Crossing the water surface should not make the player hover with
            // only their feet in the water. Sink gently until the head is under
            // water, then switch to neutral-buoyancy swimming above.
            let swim_vertical_input = (input.swim_up || input.jump) as i8 - input.swim_down as i8;
            if swim_vertical_input != 0 {
                self.velocity.y = approach(
                    self.velocity.y,
                    swim_vertical_input as f64 * self.config.swim_vertical_speed,
                    self.config.swim_vertical_acceleration * dt,
                );
            } else {
                self.velocity.y = (self.velocity.y + self.config.water_gravity * dt)
                    .max(self.config.water_terminal_velocity);
            }
            self.on_ground = false;
        } else {
            if input.jump && was_on_ground {
                self.velocity.y = self.config.jump_speed;
                self.on_ground = false;
            }
            self.velocity.y =
                (self.velocity.y + self.config.gravity * dt).max(self.config.terminal_velocity);
        }

        result.collided_x = self.move_axis(world, Axis::X, self.velocity.x * dt);
        result.collided_z = self.move_axis(world, Axis::Z, self.velocity.z * dt);
        self.on_ground = false;
        let vertical_velocity = self.velocity.y;
        result.collided_y = self.move_axis(world, Axis::Y, self.velocity.y * dt);
        if result.collided_y {
            result.landed = vertical_velocity < 0.0;
            result.hit_ceiling = vertical_velocity > 0.0;
            self.on_ground = result.landed;
        }
        result.landed |= self.on_ground;
        result
    }

    fn move_axis<W: VoxelWorld>(&mut self, world: &mut W, axis: Axis, delta: f64) -> bool {
        if delta.abs() <= COLLISION_EPSILON {
            return false;
        }

        let old = self.position;
        set_axis(&mut self.position, axis, axis_value(old, axis) + delta);
        let proposed_aabb = self.aabb();
        if !collides(world, proposed_aabb) {
            return false;
        }

        let mut resolved = axis_value(self.position, axis);
        let moving_positive = delta > 0.0;
        for_each_overlapping_block(world, proposed_aabb, |block_x, block_y, block_z| {
            let block_min = match axis {
                Axis::X => block_x as f64,
                Axis::Y => block_y as f64,
                Axis::Z => block_z as f64,
            };
            let block_max = block_min + 1.0;
            let player_min_offset = self.axis_min_offset(axis);
            let player_max_offset = self.axis_max_offset(axis);
            let candidate = if moving_positive {
                block_min - player_max_offset
            } else {
                block_max - player_min_offset
            };

            if moving_positive {
                resolved = resolved.min(candidate);
            } else {
                resolved = resolved.max(candidate);
            }
        });

        set_axis(&mut self.position, axis, resolved);
        match axis {
            Axis::X => self.velocity.x = 0.0,
            Axis::Y => self.velocity.y = 0.0,
            Axis::Z => self.velocity.z = 0.0,
        }
        true
    }

    fn axis_min_offset(&self, axis: Axis) -> f64 {
        match axis {
            Axis::X | Axis::Z => -self.config.width * 0.5,
            Axis::Y => 0.0,
        }
    }

    fn axis_max_offset(&self, axis: Axis) -> f64 {
        match axis {
            Axis::X | Axis::Z => self.config.width * 0.5,
            Axis::Y => self.config.height,
        }
    }
}

#[derive(Clone, Copy)]
enum Axis {
    X,
    Y,
    Z,
}

fn axis_value(value: DVec3, axis: Axis) -> f64 {
    match axis {
        Axis::X => value.x,
        Axis::Y => value.y,
        Axis::Z => value.z,
    }
}

fn set_axis(value: &mut DVec3, axis: Axis, axis_value: f64) {
    match axis {
        Axis::X => value.x = axis_value,
        Axis::Y => value.y = axis_value,
        Axis::Z => value.z = axis_value,
    }
}

fn approach(current: f64, target: f64, amount: f64) -> f64 {
    if (target - current).abs() <= amount {
        target
    } else {
        current + (target - current).signum() * amount
    }
}

fn block_range(min: f64, max: f64) -> std::ops::RangeInclusive<i64> {
    let first = min.floor() as i64;
    let last = (max - COLLISION_EPSILON).floor() as i64;
    first..=last
}

fn for_each_overlapping_block<W: VoxelWorld>(
    world: &mut W,
    aabb: Aabb,
    mut f: impl FnMut(i64, i64, i64),
) {
    for x in block_range(aabb.min.x, aabb.max.x) {
        for y in block_range(aabb.min.y, aabb.max.y) {
            for z in block_range(aabb.min.z, aabb.max.z) {
                if world.block_at(x, y, z).is_solid() && aabb.overlaps_block(x, y, z) {
                    f(x, y, z);
                }
            }
        }
    }
}

fn collides<W: VoxelWorld>(world: &mut W, aabb: Aabb) -> bool {
    let mut hit = false;
    for_each_overlapping_block(world, aabb, |_, _, _| hit = true);
    hit
}

fn intersects_fluid<W: VoxelWorld>(world: &mut W, aabb: Aabb) -> bool {
    for x in block_range(aabb.min.x, aabb.max.x) {
        for y in block_range(aabb.min.y, aabb.max.y) {
            for z in block_range(aabb.min.z, aabb.max.z) {
                if world.block_at(x, y, z).is_fluid() && aabb.overlaps_block(x, y, z) {
                    return true;
                }
            }
        }
    }
    false
}

fn head_intersects_fluid<W: VoxelWorld>(world: &mut W, aabb: Aabb) -> bool {
    let head_y = (aabb.max.y - COLLISION_EPSILON).floor() as i64;
    for x in block_range(aabb.min.x, aabb.max.x) {
        for z in block_range(aabb.min.z, aabb.max.z) {
            if world.block_at(x, head_y, z).is_fluid()
                && aabb.max.x > x as f64 + COLLISION_EPSILON
                && aabb.min.x < x as f64 + 1.0 - COLLISION_EPSILON
                && aabb.max.z > z as f64 + COLLISION_EPSILON
                && aabb.min.z < z as f64 + 1.0 - COLLISION_EPSILON
            {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::world::block::Block;

    #[derive(Default)]
    struct TestWorld {
        solids: HashSet<(i64, i64, i64)>,
        fluids: HashSet<(i64, i64, i64)>,
    }

    impl TestWorld {
        fn floor() -> Self {
            let mut world = Self::default();
            for x in -8..=8 {
                for z in -8..=8 {
                    world.solids.insert((x, 0, z));
                }
            }
            world
        }
    }

    impl VoxelWorld for TestWorld {
        fn block_at(&mut self, x: i64, y: i64, z: i64) -> Block {
            if self.solids.contains(&(x, y, z)) {
                Block::Stone
            } else if self.fluids.contains(&(x, y, z)) {
                Block::Water
            } else {
                Block::Air
            }
        }
    }

    fn tick(player: &mut Player, world: &mut TestWorld, input: PlayerInput) {
        player.step(world, input, 1.0 / 60.0);
    }

    #[test]
    fn gravity_lands_on_floor_without_sinking() {
        let mut world = TestWorld::floor();
        let mut player = Player::new(DVec3::new(0.5, 5.0, 0.5));

        for _ in 0..180 {
            tick(&mut player, &mut world, PlayerInput::default());
        }

        assert!(player.on_ground);
        assert!((player.position.y - 1.0).abs() < 1.0e-9);
        assert_eq!(player.velocity.y, 0.0);
    }

    #[test]
    fn horizontal_wall_stops_player() {
        let mut world = TestWorld::floor();
        for y in 1..=3 {
            world.solids.insert((2, y, 0));
        }
        let mut player = Player::new(DVec3::new(0.5, 1.0, 0.5));

        for _ in 0..120 {
            tick(
                &mut player,
                &mut world,
                PlayerInput {
                    move_x: 1.0,
                    ..PlayerInput::default()
                },
            );
        }

        assert!(player.position.x <= 2.0 - Player::new(DVec3::ZERO).config.width * 0.5 + 1.0e-9);
        assert_eq!(player.velocity.x, 0.0);
    }

    #[test]
    fn jump_leaves_ground_and_returns() {
        let mut world = TestWorld::floor();
        let mut player = Player::new(DVec3::new(0.5, 1.0, 0.5));
        player.on_ground = true;

        tick(
            &mut player,
            &mut world,
            PlayerInput {
                jump: true,
                ..PlayerInput::default()
            },
        );
        assert!(!player.on_ground);
        assert!(player.position.y > 1.0);

        for _ in 0..180 {
            tick(&mut player, &mut world, PlayerInput::default());
        }
        assert!(player.on_ground);
        assert!((player.position.y - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn diagonal_input_is_not_faster() {
        let mut world = TestWorld::floor();
        let mut player = Player::new(DVec3::new(0.5, 1.0, 0.5));
        player.on_ground = true;
        tick(
            &mut player,
            &mut world,
            PlayerInput {
                move_x: 1.0,
                move_z: 1.0,
                ..PlayerInput::default()
            },
        );
        assert!(player.velocity.length() <= player.config.walk_speed + 1.0e-9);
    }

    #[test]
    fn water_allows_neutral_and_vertical_swimming() {
        let mut world = TestWorld::floor();
        for y in 1..=8 {
            world.fluids.insert((0, y, 0));
        }
        let mut player = Player::new(DVec3::new(0.5, 2.0, 0.5));

        for _ in 0..60 {
            tick(&mut player, &mut world, PlayerInput::default());
        }
        assert!((player.position.y - 2.0).abs() < 1.0e-9);
        assert_eq!(player.velocity.y, 0.0);

        for _ in 0..60 {
            tick(
                &mut player,
                &mut world,
                PlayerInput {
                    swim_up: true,
                    ..PlayerInput::default()
                },
            );
        }
        assert!(player.position.y > 4.0);

        for _ in 0..60 {
            tick(
                &mut player,
                &mut world,
                PlayerInput {
                    swim_down: true,
                    ..PlayerInput::default()
                },
            );
        }
        assert!(player.position.y < 4.0);
    }

    #[test]
    fn walking_into_water_sinks_below_surface() {
        let mut world = TestWorld::floor();
        world.solids.insert((0, 8, 0));
        for x in 1..=8 {
            for y in 1..=8 {
                for z in -1..=1 {
                    world.fluids.insert((x, y, z));
                }
            }
        }
        let mut player = Player::new(DVec3::new(0.5, 9.0, 0.5));
        player.on_ground = true;

        for _ in 0..60 {
            tick(
                &mut player,
                &mut world,
                PlayerInput {
                    move_x: 1.0,
                    ..PlayerInput::default()
                },
            );
        }

        assert!(player.position.x > 1.0);
        assert!(player.position.y < 8.5);
    }
}
