use containerd_shim_wasm::shim::{Cli, Config};
use containerd_shim_webgpu::WasmEdgeWebGpuShim;

fn main() {
    WasmEdgeWebGpuShim::run(Some(Config {
        default_log_level: "debug".to_string(),
        ..Default::default()
    }));
}
