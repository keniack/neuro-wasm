# containerd-shim-webgpu

`containerd-shim-webgpu` is an experimental containerd shim for running WasmEdge workloads that need real GPU compute through a small generic host ABI.

This repo only keeps the WebGPU shim and demos. The base `runwasi` support crates are vendored from the upstream `containerd/runwasi` `containerd-shim-wasm/v1.0.0` release so the dependency graph stays reproducible.

The shim uses native `wgpu` on the host. In practice that means:

- Vulkan on Linux
- Metal on macOS
- DX12 on Windows

## What It Provides

- WebGPU environment normalization through the middleware layer
- optional GPU device-path discovery for common Linux layouts
- a generic `webgpu` host import module for guest Wasm modules
- real GPU-backed `compute.dispatch` execution through native `wgpu`

## Guest ABI

Import module: `webgpu`

Functions:

- `describe_runtime(output_ptr, output_cap) -> i32`
- `execute(request_ptr, request_len, input_a_ptr, input_a_len, input_b_ptr, input_b_len, output_ptr, output_cap) -> i32`

## `compute.dispatch` Contract

`execute` currently accepts `kind = "compute.dispatch"` requests.

The request body contains:

- `entrypoint` - WGSL compute entrypoint name
- `workgroups` - dispatch dimensions
- `output_words` - output buffer size in 32-bit words
- `metadata.shader_source` - WGSL source
- `metadata.result_encoding` - `u32` or `f32`
- `metadata.params_u32` - uniform data exposed at binding `3`

The shader bind-group layout is fixed:

- binding `0` - `input_a` storage buffer
- binding `1` - `input_b` storage buffer
- binding `2` - output storage buffer
- binding `3` - uniform params buffer

The response JSON includes:

- actual backend and adapter information from `wgpu`
- dispatch dimensions and invocation count
- output buffer size and checksum
- raw `output_words`
- optional decoded `output_f32`

## Build

```terminal
cargo build -p containerd-shim-webgpu
```

Or through the workspace helper:

```terminal
make build-webgpu
```

On Linux, make sure Clang and the system libc development headers are installed before building. `wasmedge-sys` runs `bindgen` over `wasmedge.h`, and missing standard headers such as `stdbool.h` will stop the build.

## Install

```terminal
sudo make install-webgpu
```

## Host Setup

For Linux/Vulkan:

1. Install the Vulkan loader and vendor Vulkan driver.
2. Verify the host GPU with `vulkaninfo --summary`.
3. Expose the render node into the runtime, for example `/dev/dri/renderD128`.

The current containerd integration still has to be built and run on Linux because the upstream `containerd-shim-wasm` stack depends on Linux-only components such as `procfs`. The `wgpu` execution layer itself is generic and can target Metal or DX12 once the surrounding shim stack is portable.

## Examples

- [webgpu-demo](/Users/kenia/workspace/neuro-wasm/examples/webgpu-demo/README.md:1)
- [image-classification-demo](/Users/kenia/workspace/neuro-wasm/examples/image-classification-demo/README.md:1)
