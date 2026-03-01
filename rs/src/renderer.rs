use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use web_sys::{WebGl2RenderingContext, WebGlProgram, WebGlShader, WebGlUniformLocation, WebGlBuffer, WebGlTexture, WebGlFramebuffer, ImageBitmap};
use glam::{Mat4, Mat3, Vec3};

const SHADOW_WIDTH: i32 = 2048;
const SHADOW_HEIGHT: i32 = 2048;

pub struct Renderer {
    gl: WebGl2RenderingContext,
    program: WebGlProgram,
    pub meshes: HashMap<String, Mesh>, 
    textures: HashMap<String, WebGlTexture>,
    
    // Main Shader Uniforms
    u_mvp: Option<WebGlUniformLocation>,
    u_model: Option<WebGlUniformLocation>,
    u_normal_matrix: Option<WebGlUniformLocation>,
    u_color: Option<WebGlUniformLocation>,
    u_use_texture: Option<WebGlUniformLocation>,
    u_texture: Option<WebGlUniformLocation>,
    u_light_pos: Option<WebGlUniformLocation>,
    u_light_dir: Option<WebGlUniformLocation>,
    u_view_pos: Option<WebGlUniformLocation>,
    u_shadow_map: Option<WebGlUniformLocation>,
    u_light_space_matrix: Option<WebGlUniformLocation>,

    // Spotlight Uniforms
    u_flashlight_on: Option<WebGlUniformLocation>,
    u_flashlight_pos: Option<WebGlUniformLocation>,
    u_flashlight_dir: Option<WebGlUniformLocation>,
    u_flashlight_cutoff: Option<WebGlUniformLocation>,
    u_flashlight_outer_cutoff: Option<WebGlUniformLocation>,
    u_shadow_map_size: Option<WebGlUniformLocation>,

    // Shadow Mapping (Top Light)
    shadow_program: WebGlProgram,
    shadow_fbo: WebGlFramebuffer,
    shadow_map_texture: WebGlTexture,
    u_shadow_light_space: Option<WebGlUniformLocation>,
    u_shadow_model: Option<WebGlUniformLocation>,

    // Shadow Mapping (Flashlight)
    flashlight_shadow_fbo: WebGlFramebuffer,
    flashlight_shadow_map_texture: WebGlTexture,
    u_flashlight_shadow_map: Option<WebGlUniformLocation>,
    u_flashlight_space_matrix: Option<WebGlUniformLocation>,
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

        // --- Shadow Map Init ---
        let shadow_program = compile_shadow_program(&gl)?;
        let u_shadow_light_space = gl.get_uniform_location(&shadow_program, "u_light_space_matrix");
        let u_shadow_model = gl.get_uniform_location(&shadow_program, "u_model");

        let shadow_fbo = gl.create_framebuffer().ok_or("Failed to create shadow FBO")?;
        let shadow_map_texture = gl.create_texture().ok_or("Failed to create shadow texture")?;
        
        gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&shadow_map_texture));
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
            WebGl2RenderingContext::TEXTURE_2D, 
            0, 
            WebGl2RenderingContext::DEPTH_COMPONENT24 as i32, 
            SHADOW_WIDTH, 
            SHADOW_HEIGHT, 
            0, 
            WebGl2RenderingContext::DEPTH_COMPONENT, 
            WebGl2RenderingContext::UNSIGNED_INT, 
            None
        )?;
        gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_MIN_FILTER, WebGl2RenderingContext::NEAREST as i32);
        gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_MAG_FILTER, WebGl2RenderingContext::NEAREST as i32);
        gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_WRAP_S, WebGl2RenderingContext::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_WRAP_T, WebGl2RenderingContext::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_COMPARE_MODE, WebGl2RenderingContext::COMPARE_REF_TO_TEXTURE as i32);
        gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_COMPARE_FUNC, WebGl2RenderingContext::LEQUAL as i32);

        gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&shadow_fbo));
        gl.framebuffer_texture_2d(
            WebGl2RenderingContext::FRAMEBUFFER, 
            WebGl2RenderingContext::DEPTH_ATTACHMENT, 
            WebGl2RenderingContext::TEXTURE_2D, 
            Some(&shadow_map_texture), 
            0
        );
        gl.draw_buffers(&js_sys::Array::of1(&JsValue::from(WebGl2RenderingContext::NONE)));
        gl.read_buffer(WebGl2RenderingContext::NONE);
        
        if gl.check_framebuffer_status(WebGl2RenderingContext::FRAMEBUFFER) != WebGl2RenderingContext::FRAMEBUFFER_COMPLETE {
            return Err(JsValue::from_str("Shadow Framebuffer not complete"));
        }
        gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, None);

        // --- Main Program Init ---
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
        let u_shadow_map = gl.get_uniform_location(&program, "u_shadow_map");
        let u_light_space_matrix = gl.get_uniform_location(&program, "u_light_space_matrix");

        let u_flashlight_shadow_map = gl.get_uniform_location(&program, "u_flashlight_shadow_map");
        let u_flashlight_space_matrix = gl.get_uniform_location(&program, "u_flashlight_space_matrix");

        let u_flashlight_on = gl.get_uniform_location(&program, "u_flashlight_on");
        let u_flashlight_pos = gl.get_uniform_location(&program, "u_flashlight_pos");
        let u_flashlight_dir = gl.get_uniform_location(&program, "u_flashlight_dir");
        let u_flashlight_cutoff = gl.get_uniform_location(&program, "u_flashlight_cutoff");
        let u_flashlight_outer_cutoff = gl.get_uniform_location(&program, "u_flashlight_outer_cutoff");
        let u_shadow_map_size = gl.get_uniform_location(&program, "u_shadow_map_size");

        // --- Flashlight Shadow Map Init ---
        let flashlight_shadow_fbo = gl.create_framebuffer().ok_or("Failed to create flashlight shadow FBO")?;
        let flashlight_shadow_map_texture = gl.create_texture().ok_or("Failed to create flashlight shadow texture")?;
        
        gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&flashlight_shadow_map_texture));
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
            WebGl2RenderingContext::TEXTURE_2D, 
            0, 
            WebGl2RenderingContext::DEPTH_COMPONENT24 as i32, 
            SHADOW_WIDTH, 
            SHADOW_HEIGHT, 
            0, 
            WebGl2RenderingContext::DEPTH_COMPONENT, 
            WebGl2RenderingContext::UNSIGNED_INT, 
            None
        )?;
        gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_MIN_FILTER, WebGl2RenderingContext::NEAREST as i32);
        gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_MAG_FILTER, WebGl2RenderingContext::NEAREST as i32);
        gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_WRAP_S, WebGl2RenderingContext::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_WRAP_T, WebGl2RenderingContext::CLAMP_TO_EDGE as i32);
        gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_COMPARE_MODE, WebGl2RenderingContext::COMPARE_REF_TO_TEXTURE as i32);
        gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, WebGl2RenderingContext::TEXTURE_COMPARE_FUNC, WebGl2RenderingContext::LEQUAL as i32);

        gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&flashlight_shadow_fbo));
        gl.framebuffer_texture_2d(
            WebGl2RenderingContext::FRAMEBUFFER, 
            WebGl2RenderingContext::DEPTH_ATTACHMENT, 
            WebGl2RenderingContext::TEXTURE_2D, 
            Some(&flashlight_shadow_map_texture), 
            0
        );
        // Check status (reuse check from above or do it here)
        if gl.check_framebuffer_status(WebGl2RenderingContext::FRAMEBUFFER) != WebGl2RenderingContext::FRAMEBUFFER_COMPLETE {
            return Err(JsValue::from_str("Flashlight Shadow Framebuffer not complete"));
        }
        gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, None);


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
            u_shadow_map,
            u_light_space_matrix,
            u_flashlight_on,
            u_flashlight_pos,
            u_flashlight_dir,
            u_flashlight_cutoff,
            u_flashlight_outer_cutoff,
            u_shadow_map_size,
            shadow_program,
            shadow_fbo,
            shadow_map_texture,
            u_shadow_light_space,
            u_shadow_model,
            flashlight_shadow_fbo,
            flashlight_shadow_map_texture,
            u_flashlight_shadow_map,
            u_flashlight_space_matrix,
        };

        // Default primitives
        let quad_mesh = create_quad_mesh(&renderer.gl)?;
        renderer.meshes.insert("quad".to_string(), quad_mesh);

        Ok(renderer)
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.gl.drawing_buffer_width() as f32 / self.gl.drawing_buffer_height() as f32
    }

    pub fn begin_shadow_pass(&self, light_space_matrix: Mat4) {
        self.gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&self.shadow_fbo));
        self.gl.viewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
        self.gl.clear(WebGl2RenderingContext::DEPTH_BUFFER_BIT);
        self.gl.use_program(Some(&self.shadow_program));
        self.gl.uniform_matrix4fv_with_f32_array(self.u_shadow_light_space.as_ref(), false, &light_space_matrix.to_cols_array());
        
        // Cull front faces for shadows to reduce peter panning
        self.gl.cull_face(WebGl2RenderingContext::FRONT);
        self.gl.enable(WebGl2RenderingContext::CULL_FACE);
    }

    pub fn begin_flashlight_shadow_pass(&self, light_space_matrix: Mat4) {
        self.gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&self.flashlight_shadow_fbo));
        self.gl.viewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
        self.gl.clear(WebGl2RenderingContext::DEPTH_BUFFER_BIT);
        self.gl.use_program(Some(&self.shadow_program)); // Reuse shadow program
        self.gl.uniform_matrix4fv_with_f32_array(self.u_shadow_light_space.as_ref(), false, &light_space_matrix.to_cols_array());
        
        self.gl.cull_face(WebGl2RenderingContext::FRONT);
        self.gl.enable(WebGl2RenderingContext::CULL_FACE);
    }

    pub fn draw_shadow_mesh(&self, mesh_name: &str, model: Mat4) -> Result<(), JsValue> {
        if let Some(mesh) = self.meshes.get(mesh_name) {
            self.gl.uniform_matrix4fv_with_f32_array(self.u_shadow_model.as_ref(), false, &model.to_cols_array());
            self.gl.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&mesh.vbo));
            // Shadow shader only needs position (loc 0)
            self.gl.enable_vertex_attrib_array(0);
            self.gl.vertex_attrib_pointer_with_i32(0, 3, WebGl2RenderingContext::FLOAT, false, 8 * 4, 0);
            
            self.gl.bind_buffer(WebGl2RenderingContext::ELEMENT_ARRAY_BUFFER, Some(&mesh.ibo));
            self.gl.draw_elements_with_i32(WebGl2RenderingContext::TRIANGLES, mesh.index_count, WebGl2RenderingContext::UNSIGNED_SHORT, 0);
        }
        Ok(())
    }

    pub fn begin_main_pass(&self) {
        self.gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, None);
        self.gl.viewport(0, 0, self.gl.drawing_buffer_width(), self.gl.drawing_buffer_height());
        self.gl.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT | WebGl2RenderingContext::DEPTH_BUFFER_BIT);
        self.gl.use_program(Some(&self.program));
        
        // Reset culling to back faces or disable
        self.gl.cull_face(WebGl2RenderingContext::BACK);
        self.gl.disable(WebGl2RenderingContext::CULL_FACE); // Or enable if we want back-face culling normally

        // Bind Shadow Map
        self.gl.active_texture(WebGl2RenderingContext::TEXTURE1);
        self.gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&self.shadow_map_texture));
        self.gl.uniform1i(self.u_shadow_map.as_ref(), 1);
        self.gl.uniform2f(self.u_shadow_map_size.as_ref(), SHADOW_WIDTH as f32, SHADOW_HEIGHT as f32);

        // Bind Flashlight Shadow Map
        self.gl.active_texture(WebGl2RenderingContext::TEXTURE2);
        self.gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&self.flashlight_shadow_map_texture));
        self.gl.uniform1i(self.u_flashlight_shadow_map.as_ref(), 2);
    }

    pub fn set_lights(&self, light_pos: Vec3, light_dir: Vec3, view_pos: Vec3, 
        flashlight_on: bool, flashlight_pos: Vec3, flashlight_dir: Vec3) {
         self.gl.uniform3f(self.u_light_pos.as_ref(), light_pos.x, light_pos.y, light_pos.z);
         self.gl.uniform3f(self.u_light_dir.as_ref(), light_dir.x, light_dir.y, light_dir.z);
         self.gl.uniform3f(self.u_view_pos.as_ref(), view_pos.x, view_pos.y, view_pos.z);

         self.gl.uniform1i(self.u_flashlight_on.as_ref(), if flashlight_on { 1 } else { 0 });
         self.gl.uniform3f(self.u_flashlight_pos.as_ref(), flashlight_pos.x, flashlight_pos.y, flashlight_pos.z);
         self.gl.uniform3f(self.u_flashlight_dir.as_ref(), flashlight_dir.x, flashlight_dir.y, flashlight_dir.z);
         // Default values for flashlight cone (inner/outer cutoff)
         self.gl.uniform1f(self.u_flashlight_cutoff.as_ref(), (12.5f32).to_radians().cos());
         self.gl.uniform1f(self.u_flashlight_outer_cutoff.as_ref(), (17.5f32).to_radians().cos());
    }

    pub fn draw_mesh(
        &self,
        mesh_name: &str,
        model: Mat4,
        view_proj: Mat4,
        light_space_matrix: Mat4,
        flashlight_space_matrix: Mat4,
        color: [f32; 4],
        texture_name: Option<&str>
    ) -> Result<(), JsValue> {
        if let Some(mesh) = self.meshes.get(mesh_name) {
            let mvp = view_proj * model;
            let normal_matrix = Mat3::from_mat4(model).inverse().transpose();
    
            self.gl.uniform_matrix4fv_with_f32_array(self.u_mvp.as_ref(), false, &mvp.to_cols_array());
            self.gl.uniform_matrix4fv_with_f32_array(self.u_model.as_ref(), false, &model.to_cols_array());
            self.gl.uniform_matrix3fv_with_f32_array(self.u_normal_matrix.as_ref(), false, &normal_matrix.to_cols_array());
            self.gl.uniform_matrix4fv_with_f32_array(self.u_light_space_matrix.as_ref(), false, &light_space_matrix.to_cols_array());
            self.gl.uniform_matrix4fv_with_f32_array(self.u_flashlight_space_matrix.as_ref(), false, &flashlight_space_matrix.to_cols_array());
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

fn compile_shadow_program(gl: &WebGl2RenderingContext) -> Result<WebGlProgram, JsValue> {
    let vs_src = r#"#version 300 es
        layout(location=0) in vec3 a_position;
        uniform mat4 u_light_space_matrix;
        uniform mat4 u_model;
        void main() {
            gl_Position = u_light_space_matrix * u_model * vec4(a_position, 1.0);
        }
    "#;
    let fs_src = r#"#version 300 es
        precision mediump float;
        void main() {
            // Depth written automatically
        }
    "#;
    
    let vs = compile_shader(gl, WebGl2RenderingContext::VERTEX_SHADER, vs_src)?;
    let fs = compile_shader(gl, WebGl2RenderingContext::FRAGMENT_SHADER, fs_src)?;
    let program = gl.create_program().ok_or("Failed to create shadow program")?;
    gl.attach_shader(&program, &vs);
    gl.attach_shader(&program, &fs);
    gl.link_program(&program);
    if !gl.get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS).as_bool().unwrap_or(false) {
        let log = gl.get_program_info_log(&program).unwrap_or_default();
        return Err(JsValue::from_str(&format!("Shadow Program Link Error: {}", log)));
    }
    Ok(program)
}

fn compile_shader_program(gl: &WebGl2RenderingContext) -> Result<WebGlProgram, JsValue> {
    let vs_src = r#"#version 300 es
        precision mediump float; // Added precision for vertex shader

        layout(location=0) in vec3 a_position;
        layout(location=1) in vec3 a_normal;
        layout(location=2) in vec2 a_tex_coord;

        uniform mat4 u_mvp;
        uniform mat4 u_model;
        uniform mat3 u_normal_matrix;
        uniform mat4 u_light_space_matrix;
        uniform mat4 u_flashlight_space_matrix; // New uniform

        out vec3 v_normal;
        out vec3 v_frag_pos;
        out vec2 v_tex_coord;
        out vec4 v_frag_pos_light_space;
        out vec4 v_frag_pos_flashlight_space; // New output

        void main() {
            gl_Position = u_mvp * vec4(a_position, 1.0);
            v_frag_pos = vec3(u_model * vec4(a_position, 1.0));
            v_normal = u_normal_matrix * a_normal;
            v_tex_coord = a_tex_coord;
            v_frag_pos_light_space = u_light_space_matrix * vec4(v_frag_pos, 1.0);
            v_frag_pos_flashlight_space = u_flashlight_space_matrix * vec4(v_frag_pos, 1.0);
        }
    "#;

    let fs_src = r#"#version 300 es
        precision mediump float;

        uniform vec4 u_color;
        uniform int u_use_texture;
        uniform sampler2D u_texture;
        uniform mediump sampler2DShadow u_shadow_map; 
        uniform mediump sampler2DShadow u_flashlight_shadow_map; // New uniform
        uniform vec2 u_shadow_map_size; 
        
        uniform vec3 u_light_pos;
        uniform vec3 u_light_dir;
        uniform vec3 u_view_pos;

        // Flashlight uniforms
        uniform int u_flashlight_on;
        uniform vec3 u_flashlight_pos;
        uniform vec3 u_flashlight_dir;
        uniform float u_flashlight_cutoff;
        uniform float u_flashlight_outer_cutoff;

        in vec3 v_normal;
        in vec3 v_frag_pos;
        in vec2 v_tex_coord;
        in vec4 v_frag_pos_light_space;
        in vec4 v_frag_pos_flashlight_space;

        out vec4 fragColor;

        float calculate_shadow(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir, mediump sampler2DShadow shadowMap) {
            vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
            projCoords = projCoords * 0.5 + 0.5;
            
            if(projCoords.z > 1.0) return 0.0;

            float bias = max(0.005 * (1.0 - dot(normal, lightDir)), 0.0005);
            
            float shadow = 0.0;
            vec2 texelSize = 1.0 / u_shadow_map_size; 
            for(int x = -1; x <= 1; ++x) {
                for(int y = -1; y <= 1; ++y) {
                    shadow += texture(shadowMap, vec3(projCoords.xy + vec2(x, y) * texelSize, projCoords.z - bias));
                }    
            }
            shadow /= 9.0;
            
            return shadow; // Returns 1.0 for LIT, 0.0 for SHADOW
        }

        void main() {
            vec4 baseColor = u_color;
            if (u_use_texture == 1) {
                vec4 texColor = texture(u_texture, v_tex_coord);
                if (texColor.a < 0.1) discard; 
                baseColor = texColor * u_color;
            }

            vec3 norm = normalize(v_normal);
            
            // --- Directional Light (Sun) ---
            vec3 dirLightDir = normalize(u_light_pos - v_frag_pos); 
            vec3 viewDir = normalize(u_view_pos - v_frag_pos);

            vec3 ambient = 0.1 * baseColor.rgb; 
            
            // Pass the specific shadow map
            float shadow = calculate_shadow(v_frag_pos_light_space, norm, dirLightDir, u_shadow_map);

            float diffDir = max(dot(norm, dirLightDir), 0.0);
            vec3 diffuseDir = diffDir * baseColor.rgb;
            
            vec3 halfwayDirDir = normalize(dirLightDir + viewDir);
            float specDir = pow(max(dot(norm, halfwayDirDir), 0.0), 32.0);
            vec3 specularDir = vec3(0.5) * specDir; 

            vec3 combinedLight = ambient;
            // FIX: shadow is 1.0 (LIT) -> use shadow directly
            combinedLight += shadow * (diffuseDir + specularDir);

            // --- Flashlight (Spotlight) ---
            if (u_flashlight_on == 1) {
                vec3 flashLightDir = normalize(u_flashlight_pos - v_frag_pos);
                float theta = dot(flashLightDir, normalize(-u_flashlight_dir)); 
                float epsilon = u_flashlight_cutoff - u_flashlight_outer_cutoff;
                float intensity = clamp((theta - u_flashlight_outer_cutoff) / epsilon, 0.0, 1.0);

                if (intensity > 0.0) {
                     // Calculate Flashlight Shadow
                     float flash_shadow = calculate_shadow(v_frag_pos_flashlight_space, norm, flashLightDir, u_flashlight_shadow_map);

                     // Attenuation
                     float distance = length(u_flashlight_pos - v_frag_pos);
                     float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * (distance * distance)); 

                     float diffFlash = max(dot(norm, flashLightDir), 0.0);
                     vec3 diffuseFlash = diffFlash * baseColor.rgb;
                     
                     vec3 halfwayDirFlash = normalize(flashLightDir + viewDir);
                     float specFlash = pow(max(dot(norm, halfwayDirFlash), 0.0), 32.0);
                     vec3 specularFlash = vec3(0.5) * specFlash; 
                     
                     combinedLight += flash_shadow * attenuation * intensity * (diffuseFlash + specularFlash);
                }
            }
            
            fragColor = vec4(combinedLight, baseColor.a);
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
