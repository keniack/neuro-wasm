# image-classification-demo

`image-classification-demo` is a WebGPU ABI demo for the experimental `containerd-shim-webgpu-v1` shim.

The workload:

- takes the image path as the first CLI argument
- optionally takes a model path as the second CLI argument
- reads the model and image bytes inside the guest
- decodes the image and builds a small feature tensor in the guest
- calls the generic shim-provided `webgpu.execute` host import with a `compute.dispatch` request
- sends a WGSL matrix-vector shader and real tensor buffers to the host
- receives GPU-computed logits and converts them to ranked predictions

This is still a lightweight demo model, not a full ResNet or YOLO graph runner. The important part is that the logits come from a real GPU dispatch through the generic `webgpu.execute` ABI, not from synthetic host-side scoring.

## Bundled Assets

- `models/resnet50-demo.json`
- `models/yolo-demo.json`
- `images/red-apple.ppm`
- `images/ocean.ppm`
- `images/golden-retriever.ppm`

The model files contain:

- label metadata
- `input_dim`
- a weight matrix for a small linear classifier

## Build The Wasm Module

```terminal
cargo build \
  -p image-classification-demo \
  --target wasm32-wasip1
```

This produces:

```terminal
target/wasm32-wasip1/debug/image-classification-demo.wasm
```

## Build An OCI Tar

```terminal
make build-examples-oci
```

This exports a standard OCI/rootfs container image tar to:

```terminal
target/wasm32-wasip1/debug/image-classification-demo-img.tar
```

## Build And Push A Registry Image

Build the example image:

```terminal
make docker-build-image-classification-demo
```

Push it to the default `keniack` repository:

```terminal
make docker-push-image-classification-demo
```

Override the repository or tag when needed:

```terminal
make docker-push-image-classification-demo IMAGE_REPO=keniack IMAGE_TAG=v0.1.0
```

## Run Through The Shim

This workload depends on the shim-provided `webgpu` host import module, so run it through the WebGPU shim rather than the plain `wasmedge` CLI.

The Wasm module builds on macOS, but the native containerd shim currently has to be built and run on Linux.

Import the local OCI tar and run the pre-bundled demo image:

```terminal
sudo ctr images import --all-platforms target/wasm32-wasip1/debug/image-classification-demo-img.tar

sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  docker.io/keniack/image-classification-demo:local \
  image-classification-demo
```

Or pull the pushed registry image and run it directly:

```terminal
sudo ctr images pull docker.io/keniack/image-classification-demo:latest

sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  docker.io/keniack/image-classification-demo:latest \
  image-classification-demo
```

The OCI image defaults to:

```terminal
/images/red-apple.ppm /models/resnet50-demo.json
```
