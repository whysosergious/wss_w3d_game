// ... (imports)
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{WebGl2RenderingContext, WebGlProgram, WebGlShader, WebGlUniformLocation, WebGlBuffer, WebGlTexture, ImageBitmap};
use glam::{Mat4, Vec3, Quat, Mat3};
use rapier3d::prelude::*;

// --- Physics Types ---

struct PhysicsWorld {
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    
    // Player Handle
    player_handle: Option<RigidBodyHandle>,
    
    // Body Handles (Cube visual sync)
    cube_handles: Vec<RigidBodyHandle>,
}

struct GameState {
    gl: Option<WebGl2RenderingContext>,
    program: Option<WebGlProgram>,
    meshes: HashMap<String, Mesh>,
    textures: HashMap<String, WebGlTexture>,
    camera: Camera,
    input: InputState,
    
    physics: PhysicsWorld,
    
    // Uniforms
    u_mvp: Option<WebGlUniformLocation>,
    u_model: Option<WebGlUniformLocation>,
    u_normal_matrix: Option<WebGlUniformLocation>,
    u_color: Option<WebGlUniformLocation>,
    u_use_texture: Option<WebGlUniformLocation>,
    u_texture: Option<WebGlUniformLocation>,
    u_light_pos: Option<WebGlUniformLocation>,
    u_light_dir: Option<WebGlUniformLocation>,
    u_view_pos: Option<WebGlUniformLocation>,
}

struct Mesh {
    vbo: WebGlBuffer,
    ibo: WebGlBuffer,
    index_count: i32,
}

struct Camera {
    position: Vec3,
    velocity: Vec3,
    yaw: f32,
    pitch: f32,
}

#[derive(Default, Clone, Copy)]
struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
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
            player_handle: None,
            cube_handles: Vec::new(),
        }
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            gl: None,
            program: None,
            meshes: HashMap::new(),
            textures: HashMap::new(),
            camera: Camera {
                position: Vec3::new(0.0, 1.5, 5.0),
                velocity: Vec3::ZERO,
                yaw: -90.0,
                pitch: 0.0,
            },
            input: InputState::default(),
            physics: PhysicsWorld::default(),
            u_mvp: None,
            u_model: None,
            u_normal_matrix: None,
            u_color: None,
            u_use_texture: None,
            u_texture: None,
            u_light_pos: None,
            u_light_dir: None,
            u_view_pos: None,
        }
    }
}

thread_local! {
    static STATE: Rc<RefCell<GameState>> = Rc::new(RefCell::new(GameState::default()));
}

// ... (init_webgl) ...

#[wasm_bindgen]
pub fn init_webgl(canvas: web_sys::OffscreenCanvas) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let gl = canvas
        .get_context("webgl2")?
        .unwrap()
        .dyn_into::<WebGl2RenderingContext>()?;

    gl.viewport(0, 0, canvas.width() as i32, canvas.height() as i32);
    gl.clear_color(0.1, 0.1, 0.1, 1.0);
    gl.enable(WebGl2RenderingContext::DEPTH_TEST);
    gl.enable(WebGl2RenderingContext::BLEND);
    gl.blend_func(WebGl2RenderingContext::SRC_ALPHA, WebGl2RenderingContext::ONE_MINUS_SRC_ALPHA);

    let program = compile_shader_program(&gl)?;
    gl.use_program(Some(&program));

    // Get locations
    let u_mvp = gl.get_uniform_location(&program, "u_mvp");
    let u_model = gl.get_uniform_location(&program, "u_model");
    let u_normal_matrix = gl.get_uniform_location(&program, "u_normal_matrix");
    let u_color = gl.get_uniform_location(&program, "u_color");
    let u_use_texture = gl.get_uniform_location(&program, "u_use_texture");
    let u_texture = gl.get_uniform_location(&program, "u_texture");
    let u_light_pos = gl.get_uniform_location(&program, "u_light_pos");
    let u_light_dir = gl.get_uniform_location(&program, "u_light_dir");
    let u_view_pos = gl.get_uniform_location(&program, "u_view_pos");

    // Default primitives
    let quad_mesh = create_quad_mesh(&gl)?;
    // We assume cube.obj is loaded later via load_model("cube_obj")
    
    // Init Physics World
    let mut physics = PhysicsWorld::default();
    
    // 1. Static Ground Plane (Collider only)
    let ground_collider = ColliderBuilder::cuboid(100.0, 0.1, 100.0).build();
    physics.collider_set.insert(ground_collider);

    // 2. Cube 1 (Dynamic, slightly rotated)
    let cube1_body = RigidBodyBuilder::dynamic()
        .translation(vector![0.0, 1.0, 0.0])
        .rotation(vector![0.5, 0.5, 0.0])
        .build();
    let cube1_collider = ColliderBuilder::cuboid(0.5, 0.5, 0.5).restitution(0.7).build();
    let cube1_handle = physics.rigid_body_set.insert(cube1_body);
    physics.collider_set.insert_with_parent(cube1_collider, cube1_handle, &mut physics.rigid_body_set);
    physics.cube_handles.push(cube1_handle);

    // 3. Cube 2 (Dynamic, higher up, falling on Cube 1)
    let cube2_body = RigidBodyBuilder::dynamic()
        .translation(vector![0.2, 4.0, 0.1]) // Offset slightly to hit edge/roll
        .rotation(vector![0.1, 0.0, 0.2])
        .build();
    let cube2_collider = ColliderBuilder::cuboid(0.5, 0.5, 0.5).restitution(0.7).build();
    let cube2_handle = physics.rigid_body_set.insert(cube2_body);
    physics.collider_set.insert_with_parent(cube2_collider, cube2_handle, &mut physics.rigid_body_set);
    physics.cube_handles.push(cube2_handle);

    // 4. Player (Kinematic Character Controller / Dynamic Body)
    // For simplicity, let's use a dynamic capsule that we push around with forces/velocity
    // Lock rotation to keep upright
    let player_body = RigidBodyBuilder::dynamic()
        .translation(vector![0.0, 2.0, 5.0])
        .lock_rotations()
        .linear_damping(1.0) // Don't slide forever
        .build();
    let player_collider = ColliderBuilder::capsule_y(0.75, 0.5).friction(0.0).build(); // Height 1.5ish, Radius 0.5
    let player_handle = physics.rigid_body_set.insert(player_body);
    physics.collider_set.insert_with_parent(player_collider, player_handle, &mut physics.rigid_body_set);
    physics.player_handle = Some(player_handle);

    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.gl = Some(gl);
        state.program = Some(program);
        state.meshes.insert("quad".to_string(), quad_mesh);
        
        state.physics = physics;
        
        state.u_mvp = u_mvp;
        state.u_model = u_model;
        state.u_normal_matrix = u_normal_matrix;
        state.u_color = u_color;
        state.u_use_texture = u_use_texture;
        state.u_texture = u_texture;
        state.u_light_pos = u_light_pos;
        state.u_light_dir = u_light_dir;
        state.u_view_pos = u_view_pos;
    });

    Ok(())
}

// ... (load_texture, load_model, set_input_state same) ...
#[wasm_bindgen]
pub fn load_texture(name: String, bitmap: ImageBitmap) -> Result<(), JsValue> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if let Some(gl) = &state.gl {
            let texture = gl.create_texture().ok_or("Failed to create texture")?;
            gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture));
            gl.tex_image_2d_with_u32_and_u32_and_image_bitmap(
                WebGl2RenderingContext::TEXTURE_2D, 0, WebGl2RenderingContext::RGBA as i32, WebGl2RenderingContext::RGBA, WebGl2RenderingContext::UNSIGNED_BYTE, &bitmap,
            )?;
            gl.generate_mipmap(WebGl2RenderingContext::TEXTURE_2D);
            state.textures.insert(name, texture);
        }
        Ok(())
    })
}

#[wasm_bindgen]
pub fn load_model(name: String, obj_text: String) -> Result<(), JsValue> {
    let mut positions = Vec::new();
    let mut tex_coords = Vec::new();
    let mut normals = Vec::new();
    let mut final_vertices = Vec::new(); 
    let mut indices = Vec::new();
    let mut unique_vertices = HashMap::new();

    for line in obj_text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() { continue; }
        match parts[0] {
            "v" => positions.push(Vec3::new(parts[1].parse().unwrap(), parts[2].parse().unwrap(), parts[3].parse().unwrap())),
            "vt" => tex_coords.push((parts[1].parse::<f32>().unwrap(), parts[2].parse::<f32>().unwrap())),
            "vn" => normals.push(Vec3::new(parts[1].parse().unwrap(), parts[2].parse().unwrap(), parts[3].parse().unwrap())),
            "f" => {
                for part in &parts[1..] {
                    if !unique_vertices.contains_key(*part) {
                        let indices_str: Vec<&str> = part.split('/').collect();
                        let v_idx: usize = indices_str[0].parse().unwrap();
                        let vt_idx: usize = indices_str.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                        let vn_idx: usize = indices_str.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
                        let pos = positions[v_idx - 1];
                        let tex = if vt_idx > 0 { tex_coords[vt_idx - 1] } else { (0.0, 0.0) };
                        let norm = if vn_idx > 0 { normals[vn_idx - 1] } else { Vec3::Y };
                        final_vertices.extend_from_slice(&[pos.x, pos.y, pos.z, norm.x, norm.y, norm.z, tex.0, tex.1]);
                        let new_index = (final_vertices.len() / 8) - 1;
                        unique_vertices.insert(part.to_string(), new_index as u16);
                    }
                    indices.push(*unique_vertices.get(*part).unwrap());
                }
            },
            _ => {}
        }
    }

    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if let Some(gl) = &state.gl {
            let vbo = gl.create_buffer().ok_or("Failed to create VBO")?;
            gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&vbo));
            unsafe {
                let vert_array = js_sys::Float32Array::view(&final_vertices);
                gl.buffer_data_with_array_buffer_view(WebGl2RenderingContext::ARRAY_BUFFER, &vert_array, WebGl2RenderingContext::STATIC_DRAW);
            }
            let ibo = gl.create_buffer().ok_or("Failed to create IBO")?;
            gl.bind_buffer(WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER, Some(&ibo));
            unsafe {
                let idx_array = js_sys::Uint16Array::view(&indices);
                gl.buffer_data_with_array_buffer_view(WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER, &idx_array, WebGl2RenderingContext::STATIC_DRAW);
            }
            let mesh = Mesh { vbo, ibo, index_count: indices.len() as i32 };
            state.meshes.insert(name, mesh);
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
        update_physics(&mut state);

        // Update Camera from Physics Body
        if let Some(handle) = state.physics.player_handle {
            if let Some(body) = state.physics.rigid_body_set.get(handle) {
                let t = body.translation();
                state.camera.position = Vec3::new(t.x, t.y + 0.75, t.z); // Eye height
            }
        }

        let gl = state.gl.as_ref().ok_or("WebGL context lost")?;
        gl.viewport(0, 0, gl.drawing_buffer_width(), gl.drawing_buffer_height());
        gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT | WebGl2RenderingContext::DEPTH_BUFFER_BIT);

        gl.use_program(state.program.as_ref());

        // Camera Vectors
        let (sin_yaw, cos_yaw) = state.camera.yaw.to_radians().sin_cos();
        let (sin_pitch, cos_pitch) = state.camera.pitch.to_radians().sin_cos();
        let front = Vec3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize();
        let view = Mat4::look_at_rh(state.camera.position, state.camera.position + front, Vec3::Y);
        let aspect = gl.drawing_buffer_width() as f32 / gl.drawing_buffer_height() as f32;
        let projection = Mat4::perspective_rh(45.0f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = projection * view;

        // Lights
        let light_pos = Vec3::new(0.0, 10.0, 0.0);
        let light_dir = Vec3::new(0.0, -1.0, 0.0);
        gl.uniform3f(state.u_light_pos.as_ref(), light_pos.x, light_pos.y, light_pos.z);
        gl.uniform3f(state.u_light_dir.as_ref(), light_dir.x, light_dir.y, light_dir.z);
        gl.uniform3f(state.u_view_pos.as_ref(), state.camera.position.x, state.camera.position.y, state.camera.position.z);

        // --- Draw Floor ---
        // Rapier Collider is 100x0.1x100 -> Half extents means 200x0.2x200? No, half extents are args.
        // Args: 100.0, 0.1, 100.0 -> Size 200, 0.2, 200.
        // Visual Mesh for Floor needs scaling.
        let model = Mat4::from_scale_rotation_translation(Vec3::new(200.0, 0.2, 200.0), Quat::IDENTITY, Vec3::new(0.0, 0.0, 0.0));
        if state.meshes.contains_key("cube_obj") {
             draw_mesh(gl, &state, "cube_obj", model, view_proj, [0.3, 0.35, 0.3, 1.0], None)?;
        }

        // --- Draw Physics Bodies (Cubes) ---
        if state.meshes.contains_key("cube_obj") {
            for handle in &state.physics.cube_handles {
                if let Some(body) = state.physics.rigid_body_set.get(*handle) {
                    let t = body.translation();
                    let r = body.rotation();
                    let pos = Vec3::new(t.x, t.y, t.z);
                    let rot = Quat::from_xyzw(r.i, r.j, r.k, r.w);
                    
                    // Cube collider was 0.5 half-extent -> Size 1.0
                    let model = Mat4::from_scale_rotation_translation(Vec3::ONE, rot, pos);
                    draw_mesh(gl, &state, "cube_obj", model, view_proj, [0.8, 0.5, 0.2, 1.0], None)?;
                }
            }
        }

        // --- Draw Sprite (Billboard) ---
        let view_rot_inv = Mat3::from_mat4(view).transpose();
        let sprite_rotation = Quat::from_mat3(&view_rot_inv);
        let model = Mat4::from_scale_rotation_translation(Vec3::new(1.0, 1.0, 1.0), sprite_rotation, Vec3::new(2.0, 1.0, 0.0));
        draw_mesh(gl, &state, "quad", model, view_proj, [1.0, 1.0, 1.0, 1.0], Some("shelly"))?;

        // --- Draw Fox (Model) ---
        if state.meshes.contains_key("fox") {
             let model = Mat4::from_scale_rotation_translation(Vec3::new(0.05, 0.05, 0.05), Quat::from_rotation_y(1.5), Vec3::new(-2.0, 0.0, 0.0));
             draw_mesh(gl, &state, "fox", model, view_proj, [1.0, 1.0, 1.0, 1.0], Some("fox_tex"))?; 
        }

        Ok(())
    })
}

fn update_physics(state: &mut GameState) {
    let dt = 0.016; 
    
    // Apply Input Forces to Player
    if let Some(handle) = state.physics.player_handle {
        if let Some(body) = state.physics.rigid_body_set.get_mut(handle) {
            let move_speed = 5.0;
            let jump_impulse = 5.0; // Impulse for instant velocity change

            let (sin_yaw, cos_yaw) = state.camera.yaw.to_radians().sin_cos();
            let forward_flat = Vec3::new(cos_yaw, 0.0, sin_yaw).normalize();
            let right_flat = Vec3::new(-sin_yaw, 0.0, cos_yaw).normalize();

            let mut wish_dir = Vec3::ZERO;
            if state.input.forward { wish_dir += forward_flat; }
            if state.input.backward { wish_dir -= forward_flat; }
            if state.input.right { wish_dir += right_flat; }
            if state.input.left { wish_dir -= right_flat; }

            if wish_dir.length_squared() > 0.0 {
                wish_dir = wish_dir.normalize() * move_speed;
            }

            // Set horizontal velocity directly (simple character control)
            let mut lin_vel = body.linvel().clone();
            lin_vel.x = wish_dir.x;
            lin_vel.z = wish_dir.z;

            // Jump
            // Check if grounded (simple velocity check or raycast needed properly)
            // For now, simple "if near ground" check using position
            if state.input.up && body.translation().y < 1.6 { // 1.5 is collider half-height + radius
                lin_vel.y = jump_impulse;
            }

            body.set_linvel(lin_vel, true);
            body.wake_up(true);
        }
    }

    state.physics.physics_pipeline.step(
        &vector![0.0, -9.81, 0.0],
        &state.physics.integration_parameters,
        &mut state.physics.island_manager,
        &mut state.physics.broad_phase,
        &mut state.physics.narrow_phase,
        &mut state.physics.rigid_body_set,
        &mut state.physics.collider_set,
        &mut state.physics.impulse_joint_set,
        &mut state.physics.multibody_joint_set,
        &mut state.physics.ccd_solver,
        None,
        &(),
        &(),
    );
}

// ... (draw_mesh, create_quad_mesh, create_cube_mesh - helper) ...
fn create_quad_mesh(gl: &WebGl2RenderingContext) -> Result<Mesh, JsValue> {
    #[rustfmt::skip]
    let vertices: [f32; 32] = [
        -0.5, -0.5, 0.0,   0.0, 0.0, 1.0,   0.0, 0.0,
         0.5, -0.5, 0.0,   0.0, 0.0, 1.0,   1.0, 0.0,
         0.5,  0.5, 0.0,   0.0, 0.0, 1.0,   1.0, 1.0,
        -0.5,  0.5, 0.0,   0.0, 0.0, 1.0,   0.0, 1.0,
    ];
    #[rustfmt::skip]
    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];

    let vbo = gl.create_buffer().ok_or("Failed to create VBO")?;
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&vbo));
    unsafe {
        let vert_array = js_sys::Float32Array::view(&vertices);
        gl.buffer_data_with_array_buffer_view(WebGl2RenderingContext::ARRAY_BUFFER, &vert_array, WebGl2RenderingContext::STATIC_DRAW);
    }

    let ibo = gl.create_buffer().ok_or("Failed to create IBO")?;
    gl.bind_buffer(WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER, Some(&ibo));
    unsafe {
        let idx_array = js_sys::Uint16Array::view(&indices);
        gl.buffer_data_with_array_buffer_view(WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER, &idx_array, WebGl2RenderingContext::STATIC_DRAW);
    }
    Ok(Mesh { vbo, ibo, index_count: indices.len() as i32 })
}

fn compile_shader_program(gl: &WebGl2RenderingContext) -> Result<WebGlProgram, JsValue> {
    let vs_src = r#"#version 300 es
        layout(location=0) in vec3 a_position;
        layout(location=1) in vec3 a_normal;
        layout(location=2) in vec2 a_tex_coord;

        uniform mat4 u_mvp;
        uniform mat4 u_model;
        uniform mat3 u_normal_matrix;

        out vec3 v_normal;
        out vec3 v_frag_pos;
        out vec2 v_tex_coord;

        void main() {
            gl_Position = u_mvp * vec4(a_position, 1.0);
            v_frag_pos = vec3(u_model * vec4(a_position, 1.0));
            v_normal = u_normal_matrix * a_normal;
            v_tex_coord = a_tex_coord;
        }
    "#;

    let fs_src = r#"#version 300 es
        precision mediump float;

        uniform vec4 u_color;
        uniform int u_use_texture;
        uniform sampler2D u_texture;
        
        uniform vec3 u_light_pos;
        uniform vec3 u_light_dir;
        uniform vec3 u_view_pos;

        in vec3 v_normal;
        in vec3 v_frag_pos;
        in vec2 v_tex_coord;

        out vec4 fragColor;

        void main() {
            vec4 baseColor = u_color;
            if (u_use_texture == 1) {
                vec4 texColor = texture(u_texture, v_tex_coord);
                if (texColor.a < 0.1) discard; 
                baseColor = texColor * u_color;
            }

            vec3 norm = normalize(v_normal);
            vec3 lightDir = normalize(u_light_pos - v_frag_pos);
            vec3 viewDir = normalize(u_view_pos - v_frag_pos);
            vec3 halfwayDir = normalize(lightDir + viewDir);

            vec3 ambient = 0.2 * baseColor.rgb;
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * baseColor.rgb;
            float spec = pow(max(dot(norm, halfwayDir), 0.0), 32.0);
            vec3 specular = vec3(0.5) * spec; 

            float theta = dot(lightDir, normalize(-u_light_dir));
            float cutOff = 0.91; 
            float outerCutOff = 0.82; 
            float epsilon = cutOff - outerCutOff;
            float intensity = clamp((theta - outerCutOff) / epsilon, 0.0, 1.0);

            diffuse *= intensity;
            specular *= intensity;

            vec3 result = ambient + diffuse + specular;
            fragColor = vec4(result, baseColor.a);
        }
    "#;

    let vs = compile_shader(gl, WebGl2RenderingContext::VERTEX_SHADER, vs_src)?;
    let fs = compile_shader(gl, WebGl2RenderingContext::FRAGMENT_SHADER, fs_src)?;
    let program = gl.create_program().ok_or("Failed to create program")?;
    gl.attach_shader(&program, &vs);
    gl.attach_shader(&program, &fs);
    gl.link_program(&program);
    if !gl.get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS).as_bool().unwrap_or(false) {
        let log = gl.get_program_info_log(&program).unwrap_or_default();
        return Err(JsValue::from_str(&format!("Program Link Error: {}", log)));
    }
    Ok(program)
}

fn compile_shader(gl: &WebGl2RenderingContext, type_: u32, src: &str) -> Result<WebGlShader, JsValue> {
    let shader = gl.create_shader(type_).ok_or("Failed to create shader")?;
    gl.shader_source(&shader, src);
    gl.compile_shader(&shader);
    if !gl.get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS).as_bool().unwrap_or(false) {
        let log = gl.get_shader_info_log(&shader).unwrap_or_default();
        return Err(JsValue::from_str(&format!("Shader Compile Error: {}", log)));
    }
    Ok(shader)
}

fn draw_mesh(
    gl: &WebGl2RenderingContext, 
    state: &GameState, 
    mesh_name: &str, 
    model: Mat4, 
    view_proj: Mat4, 
    color: [f32; 4],
    texture_name: Option<&str>
) -> Result<(), JsValue> {
    if let Some(mesh) = state.meshes.get(mesh_name) {
        let mvp = view_proj * model;
        let normal_matrix = Mat3::from_mat4(model).inverse().transpose();

        gl.uniform_matrix4fv_with_f32_array(state.u_mvp.as_ref(), false, &mvp.to_cols_array());
        gl.uniform_matrix4fv_with_f32_array(state.u_model.as_ref(), false, &model.to_cols_array());
        gl.uniform_matrix3fv_with_f32_array(state.u_normal_matrix.as_ref(), false, &normal_matrix.to_cols_array());
        gl.uniform4f(state.u_color.as_ref(), color[0], color[1], color[2], color[3]);

        if let Some(tex_name) = texture_name {
            if let Some(texture) = state.textures.get(tex_name) {
                gl.active_texture(WebGl2RenderingContext::TEXTURE0);
                gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(texture));
                gl.uniform1i(state.u_texture.as_ref(), 0);
                gl.uniform1i(state.u_use_texture.as_ref(), 1);
            } else {
                gl.uniform1i(state.u_use_texture.as_ref(), 0);
            }
        } else {
            gl.uniform1i(state.u_use_texture.as_ref(), 0);
        }

        gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&mesh.vbo));
        let stride = 8 * 4; 
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_with_i32(0, 3, WebGl2RenderingContext::FLOAT, false, stride, 0);
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_with_i32(1, 3, WebGl2RenderingContext::FLOAT, false, stride, 3 * 4);
        gl.enable_vertex_attrib_array(2);
        gl.vertex_attrib_pointer_with_i32(2, 2, WebGl2RenderingContext::FLOAT, false, stride, 6 * 4);

        gl.bind_buffer(WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER, Some(&mesh.ibo));
        gl.draw_elements_with_i32(WebGl2RenderingContext::TRIANGLES, mesh.index_count, WebGl2RenderingContext::UNSIGNED_SHORT, 0);
    }
    Ok(())
}