
export def proj-build [] {
	cd rs ; RUSTFLAGS='--cfg getrandom_backend="wasm_js" --cfg web_sys_unstable_apis' wasm-pack build --target web --out-dir ../src/pkg;
}
