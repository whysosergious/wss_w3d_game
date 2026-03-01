use glam::Vec3;
use rapier3d::prelude::*;

pub struct PhysicsWorld {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: BroadPhase,
    pub narrow_phase: NarrowPhase,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    pub ccd_solver: CCDSolver,
    pub query_pipeline: QueryPipeline,
    
    // Player Handle
    pub player_handle: Option<RigidBodyHandle>,
    
    // Body Handles (Cube visual sync)
    pub cube_handles: Vec<RigidBodyHandle>,
}

#[derive(Default, Clone, Copy)]
pub struct InputState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub flashlight: bool,
}

#[derive(Clone, Copy)]
pub struct Camera {
    pub position: Vec3,
    pub velocity: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 1.5, 5.0),
            velocity: Vec3::ZERO,
            yaw: -90.0,
            pitch: 0.0,
        }
    }
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            player_handle: None,
            cube_handles: Vec::new(),
        }
    }
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn init(&mut self) {
        // 1. Static Ground Plane (Collider only)
        let ground_collider = ColliderBuilder::cuboid(100.0, 0.1, 100.0).build();
        self.collider_set.insert(ground_collider);

        // 2. Cube 1 (Dynamic, slightly rotated)
        let cube1_body = RigidBodyBuilder::dynamic()
            .translation(vector![0.0, 1.0, 0.0])
            .rotation(vector![0.5, 0.5, 0.0])
            .build();
        let cube1_collider = ColliderBuilder::cuboid(0.5, 0.5, 0.5).restitution(0.7).build();
        let cube1_handle = self.rigid_body_set.insert(cube1_body);
        self.collider_set.insert_with_parent(cube1_collider, cube1_handle, &mut self.rigid_body_set);
        self.cube_handles.push(cube1_handle);

        // 3. Cube 2 (Dynamic, higher up, falling on Cube 1)
        let cube2_body = RigidBodyBuilder::dynamic()
            .translation(vector![0.2, 4.0, 0.1]) // Offset slightly to hit edge/roll
            .rotation(vector![0.1, 0.0, 0.2])
            .build();
        let cube2_collider = ColliderBuilder::cuboid(0.5, 0.5, 0.5).restitution(0.7).build();
        let cube2_handle = self.rigid_body_set.insert(cube2_body);
        self.collider_set.insert_with_parent(cube2_collider, cube2_handle, &mut self.rigid_body_set);
        self.cube_handles.push(cube2_handle);

        // 4. Player (Kinematic Character Controller / Dynamic Body)
        // For simplicity, let's use a dynamic capsule that we push around with forces/velocity
        // Lock rotation to keep upright
        let player_body = RigidBodyBuilder::dynamic()
            .translation(vector![0.0, 2.0, 5.0])
            .lock_rotations()
            .linear_damping(1.0) // Don't slide forever
            .build();
        let player_collider = ColliderBuilder::capsule_y(0.75, 0.5).friction(0.0).build(); // Height 1.5ish, Radius 0.5
        let player_handle = self.rigid_body_set.insert(player_body);
        self.collider_set.insert_with_parent(player_collider, player_handle, &mut self.rigid_body_set);
        self.player_handle = Some(player_handle);
    }

    pub fn get_player_translation(&self) -> Option<Vec3> {
        self.player_handle
            .and_then(|handle| self.rigid_body_set.get(handle))
            .map(|body| {
                let t = body.translation();
                Vec3::new(t.x, t.y, t.z)
            })
    }

    pub fn step(&mut self, input: &InputState, camera: &Camera) {
        let _dt = 0.016; 
        
        self.query_pipeline.update(&self.rigid_body_set, &self.collider_set);

        // Apply Input Forces to Player
        if let Some(handle) = self.player_handle {
            // IMMUTABLE borrow for raycast
            let player_translation = self.rigid_body_set.get(handle)
                .map(|body| body.translation())
                .copied();

            let is_grounded = if let Some(player_pos) = player_translation {
                let ray_origin = Point::new(player_pos.x, player_pos.y - 1.2, player_pos.z); // Slightly above bottom
                let ray_dir = vector![0.0, -1.0, 0.0];
                let ray = Ray::new(ray_origin, ray_dir);
                let max_toi = 0.1; // Small distance to check for ground contact
                let solid = true;
                let filter = QueryFilter::new().exclude_rigid_body(handle);
    
                self.query_pipeline.cast_ray(
                    &self.rigid_body_set,
                    &self.collider_set,
                    &ray,
                    max_toi,
                    solid,
                    filter,
                ).is_some()
            } else {
                false
            };
    
            // MUTABLE borrow to update velocity
            if let Some(body) = self.rigid_body_set.get_mut(handle) {
                let move_speed = 5.0;
                let jump_impulse = 5.0; // Impulse for instant velocity change
    
                let (sin_yaw, cos_yaw) = camera.yaw.to_radians().sin_cos();
                let forward_flat = Vec3::new(cos_yaw, 0.0, sin_yaw).normalize();
                let right_flat = Vec3::new(-sin_yaw, 0.0, cos_yaw).normalize();
    
                let mut wish_dir = Vec3::ZERO;
                if input.forward { wish_dir += forward_flat; }
                if input.backward { wish_dir -= forward_flat; }
                if input.right { wish_dir += right_flat; }
                if input.left { wish_dir -= right_flat; }
    
                if wish_dir.length_squared() > 0.0 {
                    wish_dir = wish_dir.normalize() * move_speed;
                }
    
                // Set horizontal velocity directly (simple character control)
                let mut lin_vel = body.linvel().clone();
                lin_vel.x = wish_dir.x;
                lin_vel.z = wish_dir.z;
    
                // Jump
                if input.up && is_grounded {
                    lin_vel.y = jump_impulse;
                }
    
                body.set_linvel(lin_vel, true);
                body.wake_up(true);
            }
        }
    
        self.physics_pipeline.step(
            &vector![0.0, -9.81, 0.0],
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            None,
            &(),
            &(),
        );
    }
}
