# Examples

- `webgpu-demo` - sends a real WGSL vector-add kernel through the generic `webgpu.execute` dispatch API and prints the GPU-computed output tensor.
- `image-classification-demo` - loads a lightweight model and an input image, then runs a real GPU matrix-vector inference pass through the same generic dispatch API.

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

Import the OCI images:

```terminal
sudo ctr images import target/wasm32-wasip1/debug/webgpu-demo-img.tar
sudo ctr images import target/wasm32-wasip1/debug/image-classification-demo-img.tar
```

Run `webgpu-demo`:

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

Run `image-classification-demo`:

```terminal
sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  ghcr.io/containerd/runwasi/image-classification-demo:local \
  image-classification-demo
```

The image classification OCI bundle defaults to:

```terminal
/images/red-apple.ppm /models/resnet50-demo.json
```

For more detail, see:

- [examples/webgpu-demo/README.md](/Users/kenia/workspace/runwasi/examples/webgpu-demo/README.md:1)
- [examples/image-classification-demo/README.md](/Users/kenia/workspace/runwasi/examples/image-classification-demo/README.md:1)
