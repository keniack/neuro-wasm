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
make build-examples-oci
```

This exports a standard OCI/rootfs container image tar to:

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

Import the local OCI tar and start it with the WebGPU shim:

Do not append `dispatch 16` after the container name here. With `ctr run`, any extra arguments override the image entrypoint, so the shim would try to execute `dispatch` as the program. If you need to override it explicitly, pass `/webgpu-demo.wasm dispatch 16`.

```terminal
sudo ctr images import --all-platforms target/wasm32-wasip1/debug/webgpu-demo-img.tar

sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  docker.io/keniack/webgpu-demo:local \
  webgpu-demo
```

Or pull the pushed registry image and run it directly:

```terminal
sudo ctr images pull docker.io/keniack/webgpu-demo:latest

sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  docker.io/keniack/webgpu-demo:latest \
  webgpu-demo
```
