use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use web_sys::{WebGl2RenderingContext, WebGlProgram, WebGlShader, WebGlUniformLocation, WebGlBuffer, WebGlTexture, ImageBitmap};
use glam::{Mat4, Mat3, Vec3};

pub struct Renderer {
    gl: WebGl2RenderingContext,
    program: WebGlProgram,
    pub meshes: HashMap<String, Mesh>, // Meshes need to be public for now, as lib.rs accesses it for contains_key
    textures: HashMap<String, WebGlTexture>,
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

pub struct Mesh {
    pub vbo: WebGlBuffer,
    pub ibo: WebGlBuffer,
    pub index_count: i32,
}

impl Renderer {
    pub fn new(canvas: web_sys::OffscreenCanvas) -> Result<Self, JsValue> {
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

        let mut renderer = Self {
            gl,
            program,
            meshes: HashMap::new(),
            textures: HashMap::new(),
            u_mvp,
            u_model,
            u_normal_matrix,
            u_color,
            u_use_texture,
            u_texture,
            u_light_pos,
            u_light_dir,
            u_view_pos,
        };

        // Default primitives
        let quad_mesh = create_quad_mesh(&renderer.gl)?;
        renderer.meshes.insert("quad".to_string(), quad_mesh);

        Ok(renderer)
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.gl.drawing_buffer_width() as f32 / self.gl.drawing_buffer_height() as f32
    }

    pub fn begin_frame(&self) {
        self.gl.viewport(0, 0, self.gl.drawing_buffer_width(), self.gl.drawing_buffer_height());
        self.gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT | WebGl2RenderingContext::DEPTH_BUFFER_BIT);
        self.gl.use_program(Some(&self.program));
    }

    pub fn set_lights(&self, light_pos: Vec3, light_dir: Vec3, view_pos: Vec3) {
         self.gl.uniform3f(self.u_light_pos.as_ref(), light_pos.x, light_pos.y, light_pos.z);
         self.gl.uniform3f(self.u_light_dir.as_ref(), light_dir.x, light_dir.y, light_dir.z);
         self.gl.uniform3f(self.u_view_pos.as_ref(), view_pos.x, view_pos.y, view_pos.z);
    }

    pub fn draw_mesh(
        &self,
        mesh_name: &str,
        model: Mat4,
        view_proj: Mat4,
        color: [f32; 4],
        texture_name: Option<&str>
    ) -> Result<(), JsValue> {
        if let Some(mesh) = self.meshes.get(mesh_name) {
            let mvp = view_proj * model;
            let normal_matrix = Mat3::from_mat4(model).inverse().transpose();
    
            self.gl.uniform_matrix4fv_with_f32_array(self.u_mvp.as_ref(), false, &mvp.to_cols_array());
            self.gl.uniform_matrix4fv_with_f32_array(self.u_model.as_ref(), false, &model.to_cols_array());
            self.gl.uniform_matrix3fv_with_f32_array(self.u_normal_matrix.as_ref(), false, &normal_matrix.to_cols_array());
            self.gl.uniform4f(self.u_color.as_ref(), color[0], color[1], color[2], color[3]);
    
            if let Some(tex_name) = texture_name {
                if let Some(texture) = self.textures.get(tex_name) {
                    self.gl.active_texture(WebGl2RenderingContext::TEXTURE0);
                    self.gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(texture));
                    self.gl.uniform1i(self.u_texture.as_ref(), 0);
                    self.gl.uniform1i(self.u_use_texture.as_ref(), 1);
                } else {
                    self.gl.uniform1i(self.u_use_texture.as_ref(), 0);
                }
            } else {
                self.gl.uniform1i(self.u_use_texture.as_ref(), 0);
            }
    
            self.gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&mesh.vbo));
            let stride = 8 * 4; 
            self.gl.enable_vertex_attrib_array(0);
            self.gl.vertex_attrib_pointer_with_i32(0, 3, WebGl2RenderingContext::FLOAT, false, stride, 0);
            self.gl.enable_vertex_attrib_array(1);
            self.gl.vertex_attrib_pointer_with_i32(1, 3, WebGl2RenderingContext::FLOAT, false, stride, 3 * 4);
            self.gl.enable_vertex_attrib_array(2);
            self.gl.vertex_attrib_pointer_with_i32(2, 2, WebGl2RenderingContext::FLOAT, false, stride, 6 * 4);
    
            self.gl.bind_buffer(WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER, Some(&mesh.ibo));
            self.gl.draw_elements_with_i32(WebGl2RenderingContext::TRIANGLES, mesh.index_count, WebGl2RenderingContext::UNSIGNED_SHORT, 0);
        }
        Ok(())
    }

    pub fn load_texture(&mut self, name: String, bitmap: ImageBitmap) -> Result<(), JsValue> {
        let texture = self.gl.create_texture().ok_or("Failed to create texture")?;
        self.gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture));
        self.gl.tex_image_2d_with_u32_and_u32_and_image_bitmap(
            WebGl2RenderingContext::TEXTURE_2D, 0, WebGl2RenderingContext::RGBA as i32, WebGl2RenderingContext::RGBA, WebGl2RenderingContext::UNSIGNED_BYTE, &bitmap,
        )?;
        self.gl.generate_mipmap(WebGl2RenderingContext::TEXTURE_2D);
        self.textures.insert(name, texture);
        Ok(())
    }

    pub fn load_model(&mut self, name: String, obj_text: String) -> Result<(), JsValue> {
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
    
        let vbo = self.gl.create_buffer().ok_or("Failed to create VBO")?;
        self.gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&vbo));
        unsafe {
            let vert_array = js_sys::Float32Array::view(&final_vertices);
            self.gl.buffer_data_with_array_buffer_view(WebGl2RenderingContext::ARRAY_BUFFER, &vert_array, WebGl2RenderingContext::STATIC_DRAW);
        }
        let ibo = self.gl.create_buffer().ok_or("Failed to create IBO")?;
        self.gl.bind_buffer(WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER, Some(&ibo));
        unsafe {
            let idx_array = js_sys::Uint16Array::view(&indices);
            self.gl.buffer_data_with_array_buffer_view(WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER, &idx_array, WebGl2RenderingContext::STATIC_DRAW);
        }
        let mesh = Mesh { vbo, ibo, index_count: indices.len() as i32 };
        self.meshes.insert(name, mesh);
        Ok(())
    }
}

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
