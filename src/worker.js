import init, { init_webgl, draw_frame, set_input_state } from './pkg/wss_game.js';

let offscreenCanvas = null;
let isInitialized = false;

// Input state matching the structure sent from main.js
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

function animate() {
    if (offscreenCanvas && isInitialized) {
        try {
            // Send the latest input state to Rust before drawing
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

self.onmessage = async (event) => {
    const { type, canvas, width, height, state } = event.data;

    switch (type) {
        case 'init':
            offscreenCanvas = canvas;
            // Determine initial dimensions from the transferred canvas or passed data
            if (width && height) {
                offscreenCanvas.width = width;
                offscreenCanvas.height = height;
            }
            
            await init();
            
            try {
                init_webgl(offscreenCanvas);
                isInitialized = true;
                console.log("WebGL initialized in worker");
                self.requestAnimationFrame(animate);
            } catch (e) {
                console.error("Failed to initialize WebGL:", e);
            }
            break;

        case 'resize':
            if (offscreenCanvas) {
                offscreenCanvas.width = width;
                offscreenCanvas.height = height;
                // Note: Rust side usually queries canvas.width/height directly in draw_frame
                // so explicit resize call might not be needed if we check dimensions there,
                // but setting canvas properties here is required.
            }
            break;

        case 'input':
            // Update local state, which is pushed to Rust in the render loop
            // This decouples input frequency from render frequency
            Object.assign(inputState, state);
            break;
    }
};