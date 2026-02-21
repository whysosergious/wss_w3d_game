use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{WebGl2RenderingContext, WebGlProgram, WebGlShader, WebGlUniformLocation, WebGlBuffer, WebGlTexture, ImageBitmap};
use glam::{Mat4, Vec3, Quat, Mat3};

// --- Global State ---

struct GameState {
    gl: Option<WebGl2RenderingContext>,
    program: Option<WebGlProgram>,
    
    // Resources
    meshes: HashMap<String, Mesh>,
    textures: HashMap<String, WebGlTexture>,
    
    // Scene Objects
    camera: Camera,
    input: InputState,
    
    // Uniform Locations
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

// --- Exported Functions ---

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

    // Default primitive (Cube)
    let cube_mesh = create_cube_mesh(&gl)?;
    let quad_mesh = create_quad_mesh(&gl)?;
    
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.gl = Some(gl);
        state.program = Some(program);
        state.meshes.insert("cube".to_string(), cube_mesh);
        state.meshes.insert("quad".to_string(), quad_mesh);
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

#[wasm_bindgen]
pub fn load_texture(name: String, bitmap: ImageBitmap) -> Result<(), JsValue> {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if let Some(gl) = &state.gl {
            let texture = gl.create_texture().ok_or("Failed to create texture")?;
            gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture));
            
            // Upload bitmap
            gl.tex_image_2d_with_u32_and_u32_and_image_bitmap(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                WebGl2RenderingContext::RGBA as i32,
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                &bitmap,
            )?;
            
            gl.generate_mipmap(WebGl2RenderingContext::TEXTURE_2D);
            gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_MIN_FILTER, WebGl2RenderingContext::LINEAR_MIPMAP_LINEAR as i32);
            gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_MAG_FILTER, WebGl2RenderingContext::LINEAR as i32);
            
            state.textures.insert(name, texture);
        }
        Ok(())
    })
}

#[wasm_bindgen]
pub fn load_model(name: String, obj_text: String) -> Result<(), JsValue> {
    // Simple OBJ parser (Pos, Normal, UV)
    // NOTE: This parser assumes triangulated faces (f v/vt/vn v/vt/vn v/vt/vn)
    
    let mut positions = Vec::new();
    let mut tex_coords = Vec::new();
    let mut normals = Vec::new();
    
    let mut final_vertices = Vec::new(); // [pos.x, pos.y, pos.z, nx, ny, nz, u, v]
    let mut indices = Vec::new();
    let mut unique_vertices = HashMap::new(); // "v/vt/vn" -> index

    for line in obj_text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() { continue; }
        
        match parts[0] {
            "v" => positions.push(Vec3::new(parts[1].parse().unwrap(), parts[2].parse().unwrap(), parts[3].parse().unwrap())),
            "vt" => tex_coords.push((parts[1].parse::<f32>().unwrap(), parts[2].parse::<f32>().unwrap())),
            "vn" => normals.push(Vec3::new(parts[1].parse().unwrap(), parts[2].parse().unwrap(), parts[3].parse().unwrap())),
            "f" => {
                for part in &parts[1..] {
                    // key is "v/vt/vn" string
                    if !unique_vertices.contains_key(*part) {
                        let indices_str: Vec<&str> = part.split('/').collect();
                        let v_idx: usize = indices_str[0].parse().unwrap();
                        let vt_idx: usize = indices_str.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                        let vn_idx: usize = indices_str.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

                        // OBJ is 1-based
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
        
        let input = state.input;
        update_camera(&mut state.camera, &input);

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

        // Set Light Uniforms (Spot Light relative to camera/view)
        // Fixed spot light overhead for simplicity
        let light_pos = Vec3::new(0.0, 10.0, 0.0);
        let light_dir = Vec3::new(0.0, -1.0, 0.0);
        
        gl.uniform3f(state.u_light_pos.as_ref(), light_pos.x, light_pos.y, light_pos.z);
        gl.uniform3f(state.u_light_dir.as_ref(), light_dir.x, light_dir.y, light_dir.z);
        gl.uniform3f(state.u_view_pos.as_ref(), state.camera.position.x, state.camera.position.y, state.camera.position.z);

        // --- Draw Floor ---
        let model = Mat4::from_scale_rotation_translation(Vec3::new(20.0, 0.05, 20.0), Quat::IDENTITY, Vec3::new(0.0, -0.025, 0.0));
        draw_mesh(gl, &state, "cube", model, view_proj, [0.3, 0.35, 0.3, 1.0], None)?;

        // --- Draw Sprite (Billboard) ---
        // Calculate rotation to face the camera (Spherical Billboarding)
        // The View matrix transforms World -> Camera. Its upper 3x3 is the rotation R.
        // We want the object to have the same orientation as the camera (so it faces the screen).
        // The inverse of the View rotation (R^T) gives the Camera's World rotation.
        let view_rot_inv = Mat3::from_mat4(view).transpose();
        let sprite_rotation = Quat::from_mat3(&view_rot_inv);
        
        // Scale note: The quad is 1x1. Scale it as needed.
        let model = Mat4::from_scale_rotation_translation(
            Vec3::new(1.0, 1.0, 1.0), // Size 1.0
            sprite_rotation,           // Rotation (Face Camera)
            Vec3::new(2.0, 1.0, 0.0)   // Position (Raised slightly)
        );
        draw_mesh(gl, &state, "quad", model, view_proj, [1.0, 1.0, 1.0, 1.0], Some("shelly"))?;

        // --- Draw Fox (Model) ---
        if state.meshes.contains_key("fox") {
             // The model might be large or small, scale it appropriately.
             // Pixelmannen models are often unit scale or 10x, let's try 0.1 scale first.
             let model = Mat4::from_scale_rotation_translation(Vec3::new(0.05, 0.05, 0.05), Quat::from_rotation_y(1.5), Vec3::new(-2.0, 0.0, 0.0));
             draw_mesh(gl, &state, "fox", model, view_proj, [1.0, 1.0, 1.0, 1.0], Some("fox_tex"))?; 
        }

        Ok(())
    })
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
        
        // Calculate Normal Matrix (Inverse Transpose of Model top-left 3x3)
        // glam::Mat3 from Mat4 is truncation. Inverse transpose for normals.
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
        
        // Stride is 8 floats: Pos(3), Normal(3), UV(2)
        let stride = 8 * 4; 
        
        // Pos
        let pos_loc = 0; // Fixed loc
        gl.enable_vertex_attrib_array(pos_loc);
        gl.vertex_attrib_pointer_with_i32(pos_loc, 3, WebGl2RenderingContext::FLOAT, false, stride, 0);

        // Normal
        let norm_loc = 1; // Need to bind attribute location or use layout in shader
        // We will assume location 1 for normals
        gl.enable_vertex_attrib_array(norm_loc);
        gl.vertex_attrib_pointer_with_i32(norm_loc, 3, WebGl2RenderingContext::FLOAT, false, stride, 3 * 4);

        // UV
        let uv_loc = 2; // Location 2
        gl.enable_vertex_attrib_array(uv_loc);
        gl.vertex_attrib_pointer_with_i32(uv_loc, 2, WebGl2RenderingContext::FLOAT, false, stride, 6 * 4);

        gl.bind_buffer(WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER, Some(&mesh.ibo));
        gl.draw_elements_with_i32(WebGl2RenderingContext::TRIANGLES, mesh.index_count, WebGl2RenderingContext::UNSIGNED_SHORT, 0);
    }
    Ok(())
}

fn update_camera(camera: &mut Camera, input: &InputState) {
    let dt = 0.016; // Fixed step for simplicity (60fps)
    let move_speed = 5.0;
    let jump_force = 5.0;
    let gravity = -9.8;
    let player_height = 1.5;

    // --- Orientation ---
    let (sin_yaw, cos_yaw) = camera.yaw.to_radians().sin_cos();
    let forward_flat = Vec3::new(cos_yaw, 0.0, sin_yaw).normalize();
    let right_flat = Vec3::new(-sin_yaw, 0.0, cos_yaw).normalize();

    // --- Horizontal Movement (Input -> Velocity) ---
    // Simple arcade movement: set velocity directly based on input
    let mut wish_dir = Vec3::ZERO;
    if input.forward { wish_dir += forward_flat; }
    if input.backward { wish_dir -= forward_flat; }
    if input.right { wish_dir += right_flat; }
    if input.left { wish_dir -= right_flat; }
    
    if wish_dir.length_squared() > 0.0 {
        wish_dir = wish_dir.normalize();
    }

    camera.velocity.x = wish_dir.x * move_speed;
    camera.velocity.z = wish_dir.z * move_speed;

    // --- Vertical Movement (Gravity & Jump) ---
    
    // Check Grounded
    let is_grounded = camera.position.y <= player_height + 0.001; // Epsilon

    // Apply Gravity
    camera.velocity.y += gravity * dt;

    // Jump Input
    if input.up && is_grounded {
        camera.velocity.y = jump_force;
    }

    // --- Integration ---
    camera.position += camera.velocity * dt;

    // --- Collision Resolution (Floor) ---
    if camera.position.y < player_height {
        camera.position.y = player_height;
        camera.velocity.y = 0.0;
    }
}

fn create_quad_mesh(gl: &WebGl2RenderingContext) -> Result<Mesh, JsValue> {
    // 2D Quad on XY plane, centered at 0,0, Z=0. Size 1x1.
    // Order: Pos(3), Normal(3), UV(2)
    // Looking down -Z, so Normal is +Z (0,0,1)
    
    #[rustfmt::skip]
    let vertices: [f32; 32] = [
        // Bottom-Left
        -0.5, -0.5, 0.0,   0.0, 0.0, 1.0,   0.0, 0.0,
        // Bottom-Right
         0.5, -0.5, 0.0,   0.0, 0.0, 1.0,   1.0, 0.0,
        // Top-Right
         0.5,  0.5, 0.0,   0.0, 0.0, 1.0,   1.0, 1.0,
        // Top-Left
        -0.5,  0.5, 0.0,   0.0, 0.0, 1.0,   0.0, 1.0,
    ];

    #[rustfmt::skip]
    let indices: [u16; 6] = [
        0, 1, 2, 
        0, 2, 3
    ];

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

fn create_cube_mesh(gl: &WebGl2RenderingContext) -> Result<Mesh, JsValue> {
    // 8 vertices per face (Pos + Normal + UV) for sharp edges
    // Simplified cube: 24 vertices
    // Data: x,y,z, nx,ny,nz, u,v
    
    // Helper to push quad
    let mut vertices: Vec<f32> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();
    let mut idx = 0;

    // Define 6 faces (Normal, Points)
    let faces = [
        (Vec3::Z, [-0.5,-0.5,0.5], [0.5,-0.5,0.5], [0.5,0.5,0.5], [-0.5,0.5,0.5]), // Front
        (Vec3::NEG_Z, [0.5,-0.5,-0.5], [-0.5,-0.5,-0.5], [-0.5,0.5,-0.5], [0.5,0.5,-0.5]), // Back
        (Vec3::Y, [-0.5,0.5,0.5], [0.5,0.5,0.5], [0.5,0.5,-0.5], [-0.5,0.5,-0.5]), // Top
        (Vec3::NEG_Y, [-0.5,-0.5,-0.5], [0.5,-0.5,-0.5], [0.5,-0.5,0.5], [-0.5,-0.5,0.5]), // Bottom
        (Vec3::X, [0.5,-0.5,0.5], [0.5,-0.5,-0.5], [0.5,0.5,-0.5], [0.5,0.5,0.5]), // Right
        (Vec3::NEG_X, [-0.5,-0.5,-0.5], [-0.5,-0.5,0.5], [-0.5,0.5,0.5], [-0.5,0.5,-0.5]), // Left
    ];

    for (n, p0, p1, p2, p3) in faces {
        // Quad vertices (0, 1, 2, 3)
        // p0: 0,0; p1: 1,0; p2: 1,1; p3: 0,1
        let pts = [p0, p1, p2, p3];
        let uvs = [(0.0,0.0), (1.0,0.0), (1.0,1.0), (0.0,1.0)];
        
        for i in 0..4 {
            vertices.extend_from_slice(&[
                pts[i][0] as f32, pts[i][1] as f32, pts[i][2] as f32, // Pos
                n.x, n.y, n.z,                                         // Normal
                uvs[i].0, uvs[i].1                                     // UV
            ]);
        }
        
        indices.extend_from_slice(&[idx, idx+1, idx+2, idx, idx+2, idx+3]);
        idx += 4;
    }

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
            // Texture Sample
            vec4 baseColor = u_color;
            if (u_use_texture == 1) {
                vec4 texColor = texture(u_texture, v_tex_coord);
                if (texColor.a < 0.1) discard; // Alpha test
                baseColor = texColor * u_color;
            }

            // Lighting (Blinn-Phong)
            vec3 norm = normalize(v_normal);
            vec3 lightDir = normalize(u_light_pos - v_frag_pos);
            vec3 viewDir = normalize(u_view_pos - v_frag_pos);
            vec3 halfwayDir = normalize(lightDir + viewDir);

            // Ambient
            vec3 ambient = 0.2 * baseColor.rgb;

            // Diffuse
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * baseColor.rgb;

            // Specular
            float spec = pow(max(dot(norm, halfwayDir), 0.0), 32.0);
            vec3 specular = vec3(0.5) * spec; // White specular

            // Spot Light (Simple soft edge)
            float theta = dot(lightDir, normalize(-u_light_dir));
            float cutOff = 0.91; // approx 25 deg
            float outerCutOff = 0.82; // approx 35 deg
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