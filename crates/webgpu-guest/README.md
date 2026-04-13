# webgpu-guest

`webgpu-guest` is the guest-side SDK for Wasm applications that run through `containerd-shim-webgpu`.

It hides the raw WebAssembly host imports and gives guest code a small typed API:

- `describe_runtime()`
- `execute(...)`
- `execute_raw(...)`
- `bytes_from_f32_slice(...)`
- `ComputeDispatch`
- `RuntimeDescription`
- `DispatchResponse`

## Why Use It

Without this crate, every guest application would need to declare the raw import block manually:

```rust
#[link(wasm_import_module = "webgpu")]
unsafe extern "C" {
    // ...
}
```

This crate keeps that ABI in one place and gives application code a simpler model.

## Add It To A Guest App

Inside this workspace:

```toml
[dependencies]
webgpu-guest = { workspace = true }
serde_json = { workspace = true }
```

Outside this workspace, use a path dependency until the crate is published:

```toml
[dependencies]
webgpu-guest = { path = "../../crates/webgpu-guest" }
serde_json = "1"
```

## Example

```rust
use webgpu_guest::{
    ComputeDispatch, DispatchResponse, ResultEncoding, bytes_from_f32_slice,
    describe_runtime, execute,
};

let runtime = describe_runtime()?;

let request = ComputeDispatch::new("add_vectors", shader_source, element_count)
    .workgroups([1, 1, 1])
    .params_u32([element_count as u32, 0, 0, 0])
    .result_encoding(ResultEncoding::F32)
    .metadata("label", "vector-add");

let response: DispatchResponse = execute(
    &request,
    &bytes_from_f32_slice(&input_a),
    &bytes_from_f32_slice(&input_b),
)?;
```

## Build

Compile guest applications for WASI preview 1:

```terminal
cargo build --target wasm32-wasip1
```

Then package the resulting `.wasm` file into a normal OCI/rootfs image. The current shim flow expects standard images, not a custom guest-side packaging format.
