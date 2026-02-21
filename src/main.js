class WssMain extends HTMLElement {
    constructor() {
        super();
        this.canvas = document.createElement("canvas");
        this.canvas.id = "main-canvas";
        this.canvas.style.width = "100%";
        this.canvas.style.height = "100%";
        this.canvas.style.display = "block";
        this.canvas.style.outline = "none";
        this.appendChild(this.canvas);

        this.worker = null;
        this.inputState = {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            yaw: -90.0, // Face -Z
            pitch: 0.0,
        };
    }

    connectedCallback() {
        if (this.started) return;
        this.started = true;

        // Set initial size BEFORE transferring control
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;

        this.initWorkerAndCanvas();

        window.addEventListener('resize', this.handleResize.bind(this));
        
        // Input listeners
        document.addEventListener('keydown', this.handleKey.bind(this, true));
        document.addEventListener('keyup', this.handleKey.bind(this, false));
        this.canvas.addEventListener('click', () => {
            this.canvas.requestPointerLock();
        });
        document.addEventListener('mousemove', this.handleMouseMove.bind(this));
    }

    disconnectedCallback() {
        if (this.worker) this.worker.terminate();
        // Remove listeners...
    }

    handleResize() {
        // Do NOT set this.canvas.width/height here after transfer
        if (this.worker) {
            this.worker.postMessage({
                type: 'resize',
                width: window.innerWidth,
                height: window.innerHeight
            });
        }
    }

    handleKey(isDown, e) {
        let changed = false;
        switch(e.code) {
            case 'KeyW': this.inputState.forward = isDown; changed = true; break;
            case 'KeyS': this.inputState.backward = isDown; changed = true; break;
            case 'KeyA': this.inputState.left = isDown; changed = true; break;
            case 'KeyD': this.inputState.right = isDown; changed = true; break;
            case 'Space': this.inputState.up = isDown; changed = true; break;
            case 'ShiftLeft': this.inputState.down = isDown; changed = true; break;
        }
        if (changed && this.worker) {
            this.worker.postMessage({ type: 'input', state: this.inputState });
        }
    }

    handleMouseMove(e) {
        if (document.pointerLockElement === this.canvas) {
            const sensitivity = 0.1;
            this.inputState.yaw += e.movementX * sensitivity;
            this.inputState.pitch -= e.movementY * sensitivity;
            
            // Constrain pitch
            if (this.inputState.pitch > 89.0) this.inputState.pitch = 89.0;
            if (this.inputState.pitch < -89.0) this.inputState.pitch = -89.0;

            if (this.worker) {
                this.worker.postMessage({ type: 'input', state: this.inputState });
            }
        }
    }

    async initWorkerAndCanvas() {
        if (!('OffscreenCanvas' in window)) return;

        const offscreen = this.canvas.transferControlToOffscreen();
        this.worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

        this.worker.postMessage({
            type: 'init',
            canvas: offscreen,
            width: this.canvas.width,
            height: this.canvas.height
        }, [offscreen]);
    }
}

customElements.define("wss-main", WssMain);