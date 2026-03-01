mod game;
mod renderer;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use glam::{Mat4, Vec3, Quat, Mat3};
use web_sys::ImageBitmap;

use game::{PhysicsWorld, InputState, Camera};
use renderer::Renderer;

struct GameState {
    renderer: Option<Renderer>,
    camera: Camera,
    input: InputState,
    physics: PhysicsWorld,
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            renderer: None,
            camera: Camera::default(),
            input: InputState::default(),
            physics: PhysicsWorld::default(),
        }
    }
}

thread_local! {
    static STATE: Rc<RefCell<GameState>> = Rc::new(RefCell::new(GameState::default()));
}

#[wasm_bindgen]
pub fn init_webgl(canvas: web_sys::OffscreenCanvas) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let renderer = Renderer::new(canvas)?;
    
    let mut physics = PhysicsWorld::new();
    physics.init();

    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.renderer = Some(renderer);
        state.physics = physics;
    });

    Ok(())
}

#[wasm_bindgen]
pub fn load_texture(name: String, bitmap: ImageBitmap) -> Result<(), JsValue> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if let Some(renderer) = &mut state.renderer {
            renderer.load_texture(name, bitmap)?;
        }
        Ok(())
    })
}

#[wasm_bindgen]
pub fn load_model(name: String, obj_text: String) -> Result<(), JsValue> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if let Some(renderer) = &mut state.renderer {
            renderer.load_model(name, obj_text)?;
        }
        Ok(())
    })
}

#[wasm_bindgen]
pub fn set_input_state(
    forward: bool, backward: bool, left: bool, right: bool,
    up: bool, down: bool, yaw: f32, pitch: f32
) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.input.forward = forward;
        state.input.backward = backward;
        state.input.left = left;
        state.input.right = right;
        state.input.up = up;
        state.input.down = down;
        state.camera.yaw = yaw;
        state.camera.pitch = pitch;
    });
}

#[wasm_bindgen]
pub fn draw_frame() -> Result<(), JsValue> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        
        // --- Physics Step ---
        let input = state.input;
        let camera = state.camera;
        state.physics.step(&input, &camera);

        // Update Camera from Physics Body
        if let Some(player_pos) = state.physics.get_player_translation() {
            state.camera.position = Vec3::new(player_pos.x, player_pos.y + 0.75, player_pos.z); // Eye height
        }

        let renderer = state.renderer.as_ref().ok_or("Renderer not initialized")?;
        
        // --- Shadow Pass ---
        // Light ViewProj
        let light_pos = Vec3::new(0.0, 10.0, 0.0);
        let light_dir = Vec3::new(0.0, -1.0, 0.0); // Directional light looking down
        // "Spotlight" description: positioned at top shining down. 
        // For shadow map, we need a view matrix from light pos.
        // Let's target the center of the scene (0,0,0) or similar.
        let light_target = Vec3::new(0.0, 0.0, 0.0);
        let light_view = Mat4::look_at_rh(light_pos, light_target, Vec3::X); // Up vector X? 
        // If light_dir is (0, -1, 0), Up cannot be Y. Let's use X or Z.
        
        // Projection: "Spotlight" -> Perspective? Or Directional?
        // If it's a spotlight, use perspective.
        // If it's "sun", use ortho.
        // "Spotlight positioned at the top shining down" -> Perspective 
        let light_projection = Mat4::perspective_rh(90.0f32.to_radians(), 1.0, 1.0, 50.0);
        
        let light_space_matrix = light_projection * light_view;

        renderer.begin_shadow_pass(light_space_matrix);

        // Draw Shadows (Floor usually doesn't cast shadow on itself unless complex, but let's draw everything)
        // Floor
        let model = Mat4::from_scale_rotation_translation(Vec3::new(200.0, 0.2, 200.0), Quat::IDENTITY, Vec3::new(0.0, 0.0, 0.0));
        if renderer.meshes.contains_key("cube_obj") {
             renderer.draw_shadow_mesh("cube_obj", model)?;
        }
        // Cubes
        if renderer.meshes.contains_key("cube_obj") {
            for handle in &state.physics.cube_handles {
                if let Some(body) = state.physics.rigid_body_set.get(*handle) {
                    let t = body.translation();
                    let r = body.rotation();
                    let pos = Vec3::new(t.x, t.y, t.z);
                    let rot = Quat::from_xyzw(r.i, r.j, r.k, r.w);
                    let model = Mat4::from_scale_rotation_translation(Vec3::ONE, rot, pos);
                    renderer.draw_shadow_mesh("cube_obj", model)?;
                }
            }
        }
        // Player? (Sprite billboard shadows are weird, but let's skip for now or draw as quad)
        // Fox
        if renderer.meshes.contains_key("fox") {
             let model = Mat4::from_scale_rotation_translation(Vec3::new(0.05, 0.05, 0.05), Quat::from_rotation_y(1.5), Vec3::new(-2.0, 0.0, 0.0));
             renderer.draw_shadow_mesh("fox", model)?; 
        }

        // --- Main Pass ---
        renderer.begin_main_pass();

        // Camera Vectors
        let (sin_yaw, cos_yaw) = state.camera.yaw.to_radians().sin_cos();
        let (sin_pitch, cos_pitch) = state.camera.pitch.to_radians().sin_cos();
        let front = Vec3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize();
        let view = Mat4::look_at_rh(state.camera.position, state.camera.position + front, Vec3::Y);
        let aspect = renderer.aspect_ratio();
        let projection = Mat4::perspective_rh(45.0f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = projection * view;

        // Lights
        renderer.set_lights(light_pos, light_dir, state.camera.position);

        // --- Draw Floor ---
        let model = Mat4::from_scale_rotation_translation(Vec3::new(200.0, 0.2, 200.0), Quat::IDENTITY, Vec3::new(0.0, 0.0, 0.0));
        if renderer.meshes.contains_key("cube_obj") {
             renderer.draw_mesh("cube_obj", model, view_proj, light_space_matrix, [0.3, 0.35, 0.3, 1.0], None)?;
        }

        // --- Draw Physics Bodies (Cubes) ---
        if renderer.meshes.contains_key("cube_obj") {
            for handle in &state.physics.cube_handles {
                if let Some(body) = state.physics.rigid_body_set.get(*handle) {
                    let t = body.translation();
                    let r = body.rotation();
                    let pos = Vec3::new(t.x, t.y, t.z);
                    let rot = Quat::from_xyzw(r.i, r.j, r.k, r.w);
                    
                    // Cube collider was 0.5 half-extent -> Size 1.0
                    let model = Mat4::from_scale_rotation_translation(Vec3::ONE, rot, pos);
                    renderer.draw_mesh("cube_obj", model, view_proj, light_space_matrix, [0.8, 0.5, 0.2, 1.0], None)?;
                }
            }
        }

        // --- Draw Sprite (Billboard) ---
        // Billboards might not cast/receive shadows correctly if they rotate to face camera.
        let view_rot_inv = Mat3::from_mat4(view).transpose();
        let sprite_rotation = Quat::from_mat3(&view_rot_inv);
        let model = Mat4::from_scale_rotation_translation(Vec3::new(1.0, 1.0, 1.0), sprite_rotation, Vec3::new(2.0, 1.0, 0.0));
        renderer.draw_mesh("quad", model, view_proj, light_space_matrix, [1.0, 1.0, 1.0, 1.0], Some("shelly"))?;

        // --- Draw Fox (Model) ---
        if renderer.meshes.contains_key("fox") {
             let model = Mat4::from_scale_rotation_translation(Vec3::new(0.05, 0.05, 0.05), Quat::from_rotation_y(1.5), Vec3::new(-2.0, 0.0, 0.0));
             renderer.draw_mesh("fox", model, view_proj, light_space_matrix, [1.0, 1.0, 1.0, 1.0], Some("fox_tex"))?; 
        }

        Ok(())
    })
}
