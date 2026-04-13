# webgpu-demo

`webgpu-demo` is a small WASI workload for the experimental `containerd-shim-webgpu-v1` shim.

It exercises the real GPU-backed `webgpu` ABI by:

- calling `webgpu.describe_runtime` to query the actual `wgpu` adapter/runtime
- calling `webgpu.execute` with a generic `compute.dispatch` request
- sending a WGSL vector-add shader plus two real input tensors
- printing the GPU-computed `f32` output tensor returned by the shim

## Build The Wasm Module

```terminal
cargo build --manifest-path examples/webgpu-demo/Cargo.toml --target wasm32-wasip1
```

This produces:

```terminal
target/wasm32-wasip1/debug/webgpu-demo.wasm
```

## Build An OCI Tar

```terminal
cargo build \
  --manifest-path examples/webgpu-demo/Cargo.toml \
  --target wasm32-wasip1 \
  --features oci-v1-tar
```

This also produces:

```terminal
target/wasm32-wasip1/debug/webgpu-demo-img.tar
```

## Build And Push A Registry Image

Build the example image:

```terminal
make docker-build-webgpu-demo
```

Push it to the default `keniack` repository:

```terminal
make docker-push-webgpu-demo
```

Override the repository or tag when needed:

```terminal
make docker-push-webgpu-demo IMAGE_REPO=keniack IMAGE_TAG=v0.1.0
```

## Run Through The Shim

This workload depends on the shim-provided `webgpu` host import module, so run it through the WebGPU shim rather than the plain `wasmedge` CLI.

The Wasm module builds on macOS, but the native containerd shim currently has to be built and run on Linux.

Import the OCI tar and start it with the WebGPU shim:

```terminal
sudo ctr images import target/wasm32-wasip1/debug/webgpu-demo-img.tar

sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  ghcr.io/containerd/runwasi/webgpu-demo:local \
  webgpu-demo dispatch 16
```
