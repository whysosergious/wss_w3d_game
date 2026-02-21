use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{WebGl2RenderingContext, WebGlProgram, WebGlShader, WebGlUniformLocation, WebGlBuffer};
use glam::{Mat4, Vec3, Quat};

// --- Global State ---

struct GameState {
    gl: Option<WebGl2RenderingContext>,
    program: Option<WebGlProgram>,
    cube_vbo: Option<WebGlBuffer>,
    cube_ibo: Option<WebGlBuffer>,
    camera: Camera,
    input: InputState,
    u_mvp: Option<WebGlUniformLocation>,
    u_color: Option<WebGlUniformLocation>,
}

struct Camera {
    position: Vec3,
    yaw: f32,   // Degrees
    pitch: f32, // Degrees
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
            cube_vbo: None,
            cube_ibo: None,
            camera: Camera {
                position: Vec3::new(0.0, 1.5, 5.0), // Start slightly above ground, back from origin
                yaw: -90.0, // Face -Z
                pitch: 0.0,
            },
            input: InputState::default(),
            u_mvp: None,
            u_color: None,
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
    // gl.enable(WebGl2RenderingContext::CULL_FACE); // Cull back faces for performance

    let program = compile_shader_program(&gl)?;
    gl.use_program(Some(&program));

    let u_mvp = gl.get_uniform_location(&program, "u_mvp");
    let u_color = gl.get_uniform_location(&program, "u_color");

    let (cube_vbo, cube_ibo) = create_cube_buffers(&gl)?;

    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.gl = Some(gl);
        state.program = Some(program);
        state.cube_vbo = Some(cube_vbo);
        state.cube_ibo = Some(cube_ibo);
        state.u_mvp = u_mvp;
        state.u_color = u_color;
    });

    Ok(())
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
        
        // 1. Update Camera Position
        let input = state.input;
        update_camera(&mut state.camera, &input);

        // 2. Render Scene
        let gl = state.gl.as_ref().ok_or("WebGL context lost")?;
        
        // Handle resize implicitly by checking canvas size? 
        // Ideally passed in, but we can query drawing buffer size.
        gl.viewport(0, 0, gl.drawing_buffer_width(), gl.drawing_buffer_height());
        gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT | WebGl2RenderingContext::DEPTH_BUFFER_BIT);

        gl.use_program(state.program.as_ref());

        // View Matrix
        let (sin_yaw, cos_yaw) = state.camera.yaw.to_radians().sin_cos();
        let (sin_pitch, cos_pitch) = state.camera.pitch.to_radians().sin_cos();
        
        // Calculate camera direction
        let front = Vec3::new(
            cos_yaw * cos_pitch,
            sin_pitch,
            sin_yaw * cos_pitch
        ).normalize();
        
        let view = Mat4::look_at_rh(
            state.camera.position,
            state.camera.position + front,
            Vec3::Y,
        );

        // Projection Matrix
        let aspect = gl.drawing_buffer_width() as f32 / gl.drawing_buffer_height() as f32;
        let projection = Mat4::perspective_rh(45.0f32.to_radians(), aspect, 0.1, 100.0);
        
        let view_proj = projection * view;

        // Bind Cube Buffers (we use the same cube mesh for everything: floor, box, sprite)
        // Just scaled differently.
        gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, state.cube_vbo.as_ref());
        let pos_attrib = gl.get_attrib_location(state.program.as_ref().unwrap(), "a_position");
        gl.vertex_attrib_pointer_with_i32(pos_attrib as u32, 3, WebGl2RenderingContext::FLOAT, false, 0, 0);
        gl.enable_vertex_attrib_array(pos_attrib as u32);

        gl.bind_buffer(WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER, state.cube_ibo.as_ref());

        // --- Draw Floor (Plane) ---
        // Scale a cube to be flat and wide
        let model = Mat4::from_scale_rotation_translation(
            Vec3::new(20.0, 0.05, 20.0), 
            Quat::IDENTITY, 
            Vec3::new(0.0, -0.025, 0.0)
        );
        let mvp = view_proj * model;
        gl.uniform_matrix4fv_with_f32_array(state.u_mvp.as_ref(), false, &mvp.to_cols_array());
        gl.uniform4f(state.u_color.as_ref(), 0.3, 0.35, 0.3, 1.0); // Dark Grey Floor
        gl.draw_elements_with_i32(WebGl2RenderingContext::TRIANGLES, 36, WebGl2RenderingContext::UNSIGNED_SHORT, 0);

        // --- Draw Box (Cube) ---
        let model = Mat4::from_scale_rotation_translation(
            Vec3::new(1.0, 1.0, 1.0), 
            Quat::IDENTITY, 
            Vec3::new(0.0, 0.5, 0.0) // On top of floor
        );
        let mvp = view_proj * model;
        gl.uniform_matrix4fv_with_f32_array(state.u_mvp.as_ref(), false, &mvp.to_cols_array());
        gl.uniform4f(state.u_color.as_ref(), 0.8, 0.5, 0.2, 1.0); // Orange Box
        gl.draw_elements_with_i32(WebGl2RenderingContext::TRIANGLES, 36, WebGl2RenderingContext::UNSIGNED_SHORT, 0);

        // --- Draw Sprite (Quad) ---
        // A "sprite" in 3D is often a billboarded quad.
        // For simplicity, we just draw a thin box standing up.
        let model = Mat4::from_scale_rotation_translation(
            Vec3::new(0.8, 0.8, 0.05), // Thin like a card
            Quat::from_rotation_y(0.0f32.to_radians()), // Slightly angled
            Vec3::new(1.5, 0.4, 0.0) // Next to box
        );
        let mvp = view_proj * model;
        gl.uniform_matrix4fv_with_f32_array(state.u_mvp.as_ref(), false, &mvp.to_cols_array());
        gl.uniform4f(state.u_color.as_ref(), 0.2, 0.6, 1.0, 1.0); // Blue "Sprite"
        gl.draw_elements_with_i32(WebGl2RenderingContext::TRIANGLES, 36, WebGl2RenderingContext::UNSIGNED_SHORT, 0);

        Ok(())
    })
}

// --- Internal Logic ---

fn update_camera(camera: &mut Camera, input: &InputState) {
    let speed = 0.1; // Movement speed per frame

    let (sin_yaw, cos_yaw) = camera.yaw.to_radians().sin_cos();
    
    // Forward vector on XZ plane
    let forward = Vec3::new(cos_yaw, 0.0, sin_yaw).normalize();
    let right = Vec3::new(-sin_yaw, 0.0, cos_yaw).normalize();

    if input.forward { camera.position += forward * speed; }
    if input.backward { camera.position -= forward * speed; }
    if input.right { camera.position += right * speed; }
    if input.left { camera.position -= right * speed; }
    if input.up { camera.position.y += speed; }
    if input.down { camera.position.y -= speed; }
}

fn create_cube_buffers(gl: &WebGl2RenderingContext) -> Result<(WebGlBuffer, WebGlBuffer), JsValue> {
    // Unit Cube (-0.5 to 0.5)
    #[rustfmt::skip]
    let vertices: [f32; 72] = [
        // Front
        -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,   0.5,  0.5,  0.5,  -0.5,  0.5,  0.5,
        // Back
        -0.5, -0.5, -0.5,  -0.5,  0.5, -0.5,   0.5,  0.5, -0.5,   0.5, -0.5, -0.5,
        // Top
        -0.5,  0.5, -0.5,  -0.5,  0.5,  0.5,   0.5,  0.5,  0.5,   0.5,  0.5, -0.5,
        // Bottom
        -0.5, -0.5, -0.5,   0.5, -0.5, -0.5,   0.5, -0.5,  0.5,  -0.5, -0.5,  0.5,
        // Right
         0.5, -0.5, -0.5,   0.5,  0.5, -0.5,   0.5,  0.5,  0.5,   0.5, -0.5,  0.5,
        // Left
        -0.5, -0.5, -0.5,  -0.5, -0.5,  0.5,  -0.5,  0.5,  0.5,  -0.5,  0.5, -0.5,
    ];

    #[rustfmt::skip]
    let indices: [u16; 36] = [
        0, 1, 2, 0, 2, 3,       // Front
        4, 5, 6, 4, 6, 7,       // Back
        8, 9, 10, 8, 10, 11,    // Top
        12, 13, 14, 12, 14, 15, // Bottom
        16, 17, 18, 16, 18, 19, // Right
        20, 21, 22, 20, 22, 23  // Left
    ];

    let vbo = gl.create_buffer().ok_or("Failed to create VBO")?;
    gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&vbo));
    
    // Unsafe view into memory for Float32Array
    unsafe {
        let vert_array = js_sys::Float32Array::view(&vertices);
        gl.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &vert_array,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    let ibo = gl.create_buffer().ok_or("Failed to create IBO")?;
    gl.bind_buffer(WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER, Some(&ibo));
    
    // Unsafe view into memory for Uint16Array
    unsafe {
        let idx_array = js_sys::Uint16Array::view(&indices);
        gl.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER,
            &idx_array,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    Ok((vbo, ibo))
}

fn compile_shader_program(gl: &WebGl2RenderingContext) -> Result<WebGlProgram, JsValue> {
    let vs_src = r#"#version 300 es
        in vec4 a_position;
        uniform mat4 u_mvp;
        void main() {
            gl_Position = u_mvp * a_position;
        }
    "#;

    let fs_src = r#"#version 300 es
        precision mediump float;
        uniform vec4 u_color;
        out vec4 fragColor;
        void main() {
            fragColor = u_color;
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
