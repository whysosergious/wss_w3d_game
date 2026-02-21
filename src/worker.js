// ... (imports) ...
import init, { init_webgl, draw_frame, set_input_state, load_texture, load_model } from './pkg/wss_game.js';

let offscreenCanvas = null;
let isInitialized = false;

let inputState = {
    forward: false,
    backward: false,
    left: false,
    right: false,
    up: false,
    down: false,
    yaw: -90.0,
    pitch: 0.0
};

// ... (animate function) ...

async function loadAssets() {
    try {
        // Load Sprite
        const shellyResp = await fetch('../rs/assets/shelly.png');
        if (!shellyResp.ok) throw new Error("Failed to load shelly.png");
        const shellyBlob = await shellyResp.blob();
        const shellyBitmap = await createImageBitmap(shellyBlob, { imageOrientation: 'flipY' });
        load_texture("shelly", shellyBitmap);

        // Load Fox Texture
        const foxTexResp = await fetch('../rs/assets/fox.png');
        if (!foxTexResp.ok) throw new Error("Failed to load fox.png");
        const foxTexBlob = await foxTexResp.blob();
        const foxBitmap = await createImageBitmap(foxTexBlob, { imageOrientation: 'flipY' });
        load_texture("fox_tex", foxBitmap);

        // Load Fox Model
        const modelResp = await fetch('../rs/assets/fox.obj');
        if (!modelResp.ok) throw new Error("Failed to load fox.obj");
        const modelText = await modelResp.text();
        load_model("fox", modelText);
        
        // Load Cube Model (NEW)
        const cubeResp = await fetch('../rs/assets/cube.obj');
        if (!cubeResp.ok) throw new Error("Failed to load cube.obj");
        const cubeText = await cubeResp.text();
        load_model("cube_obj", cubeText);
        console.log("Loaded model: cube_obj");
        
    } catch (e) {
        console.error("Asset loading error:", e);
    }
}

// ... (onmessage handler) ...
self.onmessage = async (event) => {
    const { type, canvas, width, height, state } = event.data;

    switch (type) {
        case 'init':
            offscreenCanvas = canvas;
            if (width && height) {
                offscreenCanvas.width = width;
                offscreenCanvas.height = height;
            }
            
            await init();
            
            try {
                init_webgl(offscreenCanvas);
                isInitialized = true;
                
                // Load assets after init
                await loadAssets();
                
                self.requestAnimationFrame(animate);
            } catch (e) {
                console.error("Failed to initialize WebGL:", e);
            }
            break;
            
        // ... (rest of cases)
        case 'resize':
             if (offscreenCanvas) {
                offscreenCanvas.width = width;
                offscreenCanvas.height = height;
            }
            break;

        case 'input':
            Object.assign(inputState, state);
            break;
    }
};

function animate() {
    if (offscreenCanvas && isInitialized) {
        try {
            set_input_state(
                inputState.forward,
                inputState.backward,
                inputState.left,
                inputState.right,
                inputState.up,
                inputState.down,
                inputState.yaw,
                inputState.pitch
            );

            draw_frame();
        } catch (e) {
            console.error("Error in render loop:", e);
        }
    }
    self.requestAnimationFrame(animate);
}
