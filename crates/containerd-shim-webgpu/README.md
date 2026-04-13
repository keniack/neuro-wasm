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
- shim-owned `model.detect` execution for object detection workloads

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

## `model.detect` Contract

`execute` also accepts `kind = "model.detect"` requests.

The request metadata contains:

- `model_path` - model path as seen from the guest, for example `examples/yolo-detection-demo/models/model.onnx` or `/models/yolov8l.onnx`
- `task` - currently `object-detection`
- `score_threshold`
- `iou_threshold`
- `max_detections`
- `provider` - optional ONNX Runtime execution provider override

The guest passes image bytes in `input_a`. The shim resolves `model_path` on the host
and executes detection in the broker, returning JSON detections to the guest.

Model path resolution order:

1. `bundle/rootfs/<model_path>` — the file is bundled inside the container image.
2. `WEBGPU_MODEL_DIR/<model_path>` — the file lives on the host at the directory configured by `WEBGPU_MODEL_DIR`.

`WEBGPU_MODEL_DIR` is a host-only env var consumed by the shim and never forwarded into the guest environment.

Model formats:

- `.json` - lightweight demo model executed through the existing host-side `wgpu` path
- `.onnx` - ONNX model executed through a host-side ONNX Runtime helper

`.onnx` support requires the machine running `containerd-shim-webgpu-v1` to have:

- `python3`
- `onnxruntime`
- `numpy`
- `Pillow`

If `provider` is omitted, the shim-side runner will choose the first suitable
ONNX Runtime execution provider available on the host and fall back to CPU.

## Build

```terminal
cargo build -p containerd-shim-webgpu
```

Or through the workspace helper:

```terminal
make build-webgpu
```

On Debian/Ubuntu you can install the common build prerequisites with:

```terminal
sudo make install-build-deps-debian
```

That covers:

- `build-essential`
- `clang`
- `libclang-dev`
- `libc6-dev`
- `libseccomp-dev`
- `vulkan-tools`
- `libvulkan1`

On Linux, make sure Clang, the system libc development headers, and the `libseccomp` development package are installed before building. `wasmedge-sys` runs `bindgen` over `wasmedge.h`, and missing standard headers such as `stdbool.h` will stop the build. The vendored `runwasi` shim stack also links against `libseccomp`.

The default build uses the standalone dynamic WasmEdge library. If you explicitly need static WasmEdge linking, enable the `static` feature yourself; that typically also requires extra system linker dependencies such as `libzstd` and the C++ runtime.

## Install

```terminal
make build-webgpu
sudo make install-webgpu
```

## Host Setup

For Linux/Vulkan:

1. Install the Vulkan loader and vendor Vulkan driver.
2. Verify the host GPU with `vulkaninfo --summary`.
3. Expose the render node into the runtime, for example `/dev/dri/renderD128`.

The shim keeps the native GPU stack on the host side and brokers WebGPU requests from the guest over an internal Unix socket. `scratch` Wasm images do not need host Vulkan libraries inside the container, and host-only settings such as `WEBGPU_DEVICE_PATH` and `WEBGPU_MODEL_DIR` stay in the shim instead of being forwarded into the guest Wasm environment. If the shim still logs `libvulkan.so.1: cannot open shared object file` or `missing Vulkan entry points`, the Vulkan loader is missing on the host. Install `libvulkan1` and the vendor driver on the machine running `containerd`.

The current containerd integration still has to be built and run on Linux because the upstream `containerd-shim-wasm` stack depends on Linux-only components such as `procfs`. The `wgpu` execution layer itself is generic and can target Metal or DX12 once the surrounding shim stack is portable.

## Debugging

If `ctr run` fails before guest execution starts, inspect the host containerd logs:

```terminal
sudo journalctl -u containerd -f
```

The shim binary defaults to `debug` log level, so startup validation failures should appear in the containerd journal without extra logger setup.

The vendored `runwasi` shim now logs workload classification details, including OCI wasm layer detection, resolved entrypoint candidates, and the exact reason behind `executor can't handle spec` failures.

## Examples

- [webgpu-demo](../../examples/webgpu-demo/README.md)
- [image-classification-demo](../../examples/image-classification-demo/README.md)
- [yolo-detection-demo](../../examples/yolo-detection-demo/README.md)
