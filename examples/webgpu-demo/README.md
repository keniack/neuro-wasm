# webgpu-demo

`webgpu-demo` is a small WASI workload for the experimental `containerd-shim-webgpu-v1` shim.

It exercises the real GPU-backed `webgpu` ABI by:

- calling `webgpu.describe_runtime` to query the actual `wgpu` adapter/runtime
- calling `webgpu.execute` with a generic `compute.dispatch` request
- sending a WGSL vector-add shader plus two real input tensors
- printing the GPU-computed `f32` output tensor returned by the shim
- printing compact dispatch metadata rather than echoing the full WGSL source back to stdout

The guest code uses the workspace `webgpu_guest` helper crate, so it does not need to declare the raw WebAssembly imports manually. The current example uses `ComputeDispatch`, `execute(...)`, and `bytes_from_f32_slice(...)`.

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

Use the explicit Wasm path form with `ctr run`. That makes the intended guest argv unambiguous: `["/webgpu-demo.wasm", "dispatch", "16"]`.

`dispatch 16` is specific to this demo. It means: run the `dispatch` subcommand with `16` vector elements.

```terminal
sudo ctr images import --all-platforms target/wasm32-wasip1/debug/webgpu-demo-img.tar

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
  webgpu-demo \
  /webgpu-demo.wasm dispatch 16
```

This image stays `scratch`. The WebGPU shim keeps Vulkan on the host side and brokers WebGPU requests from the guest over an internal Unix socket; host-only settings such as `WEBGPU_DEVICE_PATH` are not forwarded into the guest Wasm environment. If the shim still logs `libvulkan.so.1: cannot open shared object file` or `missing Vulkan entry points`, install the Vulkan loader on the host that runs `containerd-shim-webgpu-v1`, for example `sudo apt-get install libvulkan1 vulkan-tools`, then check `vulkaninfo --summary`.

On success, the output should include:

- `webgpu.runtime_ready=true`
- a real host adapter name such as `NVIDIA RTX A4000`
- `dispatch.output_f32=...` with the computed sums

The response metadata is summarized by the shim. Expect fields such as `label`, `source`, `shader_source_bytes`, and `params_u32_len` instead of the full `shader_source` string.
