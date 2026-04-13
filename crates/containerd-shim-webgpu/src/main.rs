use containerd_shim_wasm::shim::Cli;
use containerd_shim_webgpu::WasmEdgeWebGpuShim;

fn main() {
    WasmEdgeWebGpuShim::run(None);
}
