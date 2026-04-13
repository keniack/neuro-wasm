# WebGPU Shim Workspace

This workspace is focused on a real GPU-backed WebGPU-style compute shim for WasmEdge workloads.

The main runtime is [containerd-shim-webgpu](/Users/kenia/workspace/neuro-wasm/crates/containerd-shim-webgpu/README.md:1). It exposes a generic `webgpu` host import module to guest Wasm workloads:

- `describe_runtime`
- `execute`

`execute` accepts a generic `compute.dispatch` request. The guest passes WGSL shader source plus raw input buffers, and the shim runs a real native `wgpu` compute pipeline on the configured backend. The current demos use that path for:

- real vector-add on the GPU
- real tensor-style matrix-vector inference for image classification

## Workspace Layout

- [crates/containerd-shim-webgpu](/Users/kenia/workspace/neuro-wasm/crates/containerd-shim-webgpu/README.md:1) - the WebGPU shim
- [examples/webgpu-demo](/Users/kenia/workspace/neuro-wasm/examples/webgpu-demo/README.md:1) - generic WGSL dispatch smoke test
- [examples/image-classification-demo](/Users/kenia/workspace/neuro-wasm/examples/image-classification-demo/README.md:1) - image classification over a real GPU tensor dispatch

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

Build and push both example container images to the default `keniack` repository:

```terminal
make docker-push-examples
```

Override the repository or tag when needed:

```terminal
make docker-push-examples IMAGE_REPO=keniack IMAGE_TAG=v0.1.0
```

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

```terminal
sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  ghcr.io/containerd/runwasi/webgpu-demo:local \
  webgpu-demo dispatch 16
```

Useful environment variables:

- `WEBGPU_ENABLED=1|0`
- `WEBGPU_REQUIRED=1|0`
- `WEBGPU_BACKEND=auto|vulkan|metal|dx12|gl`
- `WEBGPU_DEVICE_PATH=/dev/dri/renderD128`
- `WEBGPU_ADAPTER_NAME=<adapter-name>`
- `WEBGPU_MAX_BUFFER_SIZE=<bytes>`
- `WEBGPU_MAX_BIND_GROUPS=<count>`
- `WEBGPU_FORCE_FALLBACK_ADAPTER=1|0`

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
