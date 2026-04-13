# Examples

- `webgpu-demo` - sends a real WGSL vector-add kernel through the generic `webgpu.execute` dispatch API and prints the GPU-computed output tensor.
- `image-classification-demo` - queries `webgpu.describe_runtime`, then loads a lightweight model and an input image and runs a real GPU matrix-vector inference pass through the same generic dispatch API.

## Build

Build and install the shim on Linux:

```terminal
make build-webgpu
sudo make install-webgpu
```

Build both Wasm apps:

```terminal
make build-examples
```

Build both OCI tar artifacts:

```terminal
make build-examples-oci
```

The local tar artifacts are standard OCI/rootfs container images exported through Docker. The `.wasm` entrypoint lives inside the image filesystem and the shim runs it from the mounted container rootfs; these examples are not packaged through per-example Cargo `build.rs` scripts.

Build both example container images:

```terminal
make docker-build-examples
```

Push both images to the default `keniack` repository:

```terminal
make docker-push-examples
```

Override the repository or tag when needed:

```terminal
make docker-push-examples IMAGE_REPO=keniack IMAGE_TAG=v0.1.0
```

This produces:

```terminal
target/wasm32-wasip1/debug/webgpu-demo.wasm
target/wasm32-wasip1/debug/webgpu-demo-img.tar
target/wasm32-wasip1/debug/image-classification-demo.wasm
target/wasm32-wasip1/debug/image-classification-demo-img.tar
```

The Docker targets publish:

```terminal
keniack/webgpu-demo:latest
keniack/image-classification-demo:latest
```

## Execute

Import the local OCI images built by `make build-examples-oci`:

```terminal
sudo ctr images import --all-platforms target/wasm32-wasip1/debug/webgpu-demo-img.tar
sudo ctr images import --all-platforms target/wasm32-wasip1/debug/image-classification-demo-img.tar
```

Run `webgpu-demo`:

Use the explicit Wasm path form with `ctr run`. It matches the argv the guest expects and avoids accidental entrypoint overrides.

```terminal
sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  docker.io/keniack/webgpu-demo:local \
  webgpu-demo \
  /webgpu-demo.wasm dispatch 16
```

Run `image-classification-demo`:

```terminal
sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  docker.io/keniack/image-classification-demo:local \
  image-classification-demo
```

Or pull the pushed registry images and run those directly:

```terminal
sudo ctr images pull docker.io/keniack/webgpu-demo:latest
sudo ctr images pull docker.io/keniack/image-classification-demo:latest
```

These example images stay `scratch`. The WebGPU shim keeps Vulkan on the host side and brokers WebGPU requests from the guest over an internal Unix socket; host-only settings such as `WEBGPU_DEVICE_PATH` are not forwarded into the guest Wasm environment. If the shim still logs `libvulkan.so.1: cannot open shared object file` or `missing Vulkan entry points`, install the Vulkan loader on the host that runs `containerd-shim-webgpu-v1`, for example `sudo apt-get install libvulkan1 vulkan-tools`.

The image classification OCI bundle defaults to:

```terminal
/images/red-apple.ppm /models/resnet50-demo.json
```

For more detail, see:

- [examples/webgpu-demo/README.md](/Users/kenia/workspace/neuro-wasm/examples/webgpu-demo/README.md:1)
- [examples/image-classification-demo/README.md](/Users/kenia/workspace/neuro-wasm/examples/image-classification-demo/README.md:1)
