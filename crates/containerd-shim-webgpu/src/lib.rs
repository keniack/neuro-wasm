pub mod host;
pub mod instance;
pub mod middleware;

pub use instance::WasmEdgeWebGpuShim;

#[cfg(unix)]
#[cfg(test)]
#[path = "tests.rs"]
mod wasmedge_webgpu_tests;
