# WebGPU Shim Workspace

This workspace is focused on a real GPU-backed WebGPU-style compute shim for WasmEdge workloads.

The main runtime is [containerd-shim-webgpu](crates/containerd-shim-webgpu/README.md). It exposes a generic `webgpu` host import module to guest Wasm workloads:

- `describe_runtime`
- `execute`

`execute` accepts a generic `compute.dispatch` request. The guest passes WGSL shader source plus raw input buffers, and the shim runs a real native `wgpu` compute pipeline on the configured backend. The current demos use that path for:

- real vector-add on the GPU
- real tensor-style matrix-vector inference for image classification
- shim-owned ONNX object detection with models loaded from the host

The same host import module also exposes a shim-owned `model.detect` flow through the existing `execute(...)` ABI. In that mode, the guest sends image bytes plus a model path and the shim resolves and runs the model on the host side, keeping the guest isolated. The shim looks for the model inside the container rootfs first, then falls back to the host directory configured by `WEBGPU_MODEL_DIR`.

## Workspace Layout

- [crates/containerd-shim-webgpu](crates/containerd-shim-webgpu/README.md) - the WebGPU shim
- [crates/webgpu-guest](crates/webgpu-guest/README.md) - guest-side SDK for Wasm applications
- [examples/webgpu-demo](examples/webgpu-demo/README.md) - generic WGSL dispatch smoke test
- [examples/image-classification-demo](examples/image-classification-demo/README.md) - image classification over a real GPU tensor dispatch
- [examples/yolo-detection-demo](examples/yolo-detection-demo/README.md) - shim-owned ONNX object detection loaded from the host via `WEBGPU_MODEL_DIR`

The shim support crates are vendored from the upstream `containerd/runwasi` `containerd-shim-wasm/v1.0.0` release so the dependency graph can be pinned locally instead of drifting with upstream git and crates.io resolution.

## Prerequisites

You need a native GPU backend installed on the host.

Build prerequisites on Debian/Ubuntu:

```terminal
sudo make install-build-deps-debian
```

That target installs:

- `build-essential`
- `clang`
- `libclang-dev`
- `libc6-dev`
- `libseccomp-dev`
- `vulkan-tools`
- `libvulkan1`

Equivalent Fedora/RHEL packages:

- `gcc`
- `gcc-c++`
- `clang`
- `libclang-devel`
- `glibc-devel`
- `libseccomp-devel`
- `vulkan-tools`
- `vulkan-loader`

For Linux with Vulkan:

1. Install the Vulkan loader and vendor driver stack.
2. Install a Vulkan ICD for your GPU, for example Mesa Vulkan drivers or the NVIDIA proprietary Vulkan stack.
3. Make sure the container runtime can see the GPU device nodes you want to use, for example `/dev/dri/renderD128` or `/dev/nvidia0`.
4. Verify the host can enumerate the GPU with `vulkaninfo --summary`.

The shim uses `wgpu` natively, so it talks to the host GPU stack rather than emulating compute in the guest.

The demo Wasm modules build on macOS, but the native containerd shim currently needs a Linux host because the upstream `containerd-shim-wasm` stack still depends on Linux-only components such as `procfs`.

The Linux build also needs a working C toolchain for `bindgen`. `wasmedge-sys` generates bindings from `wasmedge.h`, so Clang and the system libc development headers must be installed. The shim stack also links against `libseccomp`, so the `libseccomp` development package is required during build.

The default shim build uses the standalone dynamic WasmEdge library instead of static linking. That avoids extra linker dependencies during build, but the resulting binary needs `libwasmedge.so` available on the runtime library path unless you opt into the `static` feature yourself.

## Build

Build the shim:

```terminal
make build-webgpu
```

Build the demo Wasm modules:

```terminal
make build-examples
```

Build the demo OCI tar artifacts:

```terminal
make build-examples-oci
```

That exports standard OCI/rootfs container images to:

- `target/wasm32-wasip1/debug/webgpu-demo-img.tar`
- `target/wasm32-wasip1/debug/image-classification-demo-img.tar`
- `target/wasm32-wasip1/debug/yolo-detection-demo-img.tar`

Build and push all example container images to the default `keniack` repository:

```terminal
make docker-push-examples
```

Override the repository or tag when needed:

```terminal
make docker-push-examples IMAGE_REPO=keniack IMAGE_TAG=v0.1.0
```

## Build Guest Apps

Guest Wasm applications should use the workspace SDK crate [crates/webgpu-guest](crates/webgpu-guest/README.md) instead of declaring the raw `extern "C"` imports by hand.

Inside this workspace, add:

```toml
[dependencies]
webgpu-guest = { workspace = true }
serde_json = { workspace = true }
```

Then build requests through the SDK:

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

For shim-owned ONNX detection:

```rust
use webgpu_guest::{ModelDetect, detect};

let request = ModelDetect::new("examples/yolo-detection-demo/models/model.onnx")
    .score_threshold(0.25)
    .iou_threshold(0.45)
    .max_detections(20);

let response = detect(&request, &image_bytes)?;
```

The shim-owned host runner needs `python3`, `onnxruntime`, `numpy`, and `Pillow` installed
on the machine running `containerd-shim-webgpu-v1`. Set `WEBGPU_MODEL_DIR` on the host so
the shim can resolve the model path without bundling it in the container image.

Build the guest module with:

```terminal
cargo build --target wasm32-wasip1
```

Then package it as a normal OCI/rootfs image, for example a `scratch` image with the `.wasm` file copied to `/app.wasm` and `ENTRYPOINT ["/app.wasm"]`.

## Install

Install the shim binary into `PREFIX/bin`:

```terminal
make build-webgpu
sudo make install-webgpu
```

Default `PREFIX` is `/usr/local`.

## Run

Run workloads with the runtime name `io.containerd.webgpu.v1`.

Example on Linux/Vulkan:

With `ctr run`, the most reliable form is to pass the Wasm path explicitly after the container ID. That preserves the intended argv inside the guest and avoids confusion around `ENTRYPOINT`/`CMD` overrides.

```terminal
sudo ctr images pull docker.io/keniack/webgpu-demo:latest

sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  docker.io/keniack/webgpu-demo:latest \
  webgpu-demo \
  /webgpu-demo.wasm dispatch 16
```

The WebGPU shim keeps the native GPU stack on the host side and brokers WebGPU requests from the guest over an internal Unix socket. `scratch` images do not need host Vulkan libraries inside the container, and host-only settings such as `WEBGPU_DEVICE_PATH` are consumed by the shim instead of being forwarded into the guest Wasm environment. If the shim still logs `libvulkan.so.1: cannot open shared object file` or `missing Vulkan entry points`, the Vulkan loader or driver is missing on the host itself. Install the host package first, for example `sudo apt-get install libvulkan1 vulkan-tools`, then verify with `vulkaninfo --summary`.

Useful environment variables:

- `WEBGPU_ENABLED=1|0`
- `WEBGPU_REQUIRED=1|0`
- `WEBGPU_BACKEND=auto|vulkan|metal|dx12|gl`
- `WEBGPU_DEVICE_PATH=/dev/dri/renderD128`
- `WEBGPU_ADAPTER_NAME=<adapter-name>`
- `WEBGPU_MAX_BUFFER_SIZE=<bytes>`
- `WEBGPU_MAX_BIND_GROUPS=<count>`
- `WEBGPU_FORCE_FALLBACK_ADAPTER=1|0`
- `WEBGPU_MODEL_DIR=<host-path>` — host directory searched when a model is not found inside the container rootfs (host-only, not forwarded to the guest)

## Debugging

If `ctr run` fails during shim startup, inspect the containerd logs on the host:

```terminal
sudo journalctl -u containerd -f
```

The `containerd-shim-webgpu` binary now defaults to `debug` log level so startup classification failures are emitted without extra logger configuration.

Or, for recent shim errors only:

```terminal
sudo journalctl -u containerd --since "10 minutes ago" | grep -E "webgpu|runwasi|wasm workload probe|linux workload probe|can't handle spec"
```

The vendored shim now logs:

- whether OCI wasm layers were loaded for the container
- the entrypoint, args, module name, and resolved file candidates
- whether the runtime classified the workload as Linux or Wasm
- the concrete reason a `can't handle spec` decision was made
- the bundle path used to build the container

## Integration Contract

The real GPU path currently uses a fixed bind-group contract for `compute.dispatch`:

- binding `0`: `input_a` storage buffer
- binding `1`: `input_b` storage buffer
- binding `2`: output storage buffer
- binding `3`: uniform params buffer built from `metadata.params_u32`

The request metadata must include:

- `shader_source`: WGSL compute shader source
- `result_encoding`: `u32` or `f32`
- `params_u32`: uniform words passed to binding `3`

This is intentionally simpler than full browser WebGPU, but it executes real GPU workloads and is enough for tensor-style compute kernels.
