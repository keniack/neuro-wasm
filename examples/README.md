# Examples

- `webgpu-demo` - sends a real WGSL vector-add kernel through the generic `webgpu.execute` dispatch API and prints the GPU-computed output tensor.
- `image-classification-demo` - queries `webgpu.describe_runtime`, then loads a lightweight model and an input image and runs a real GPU matrix-vector inference pass through the same generic dispatch API.
- `yolo-detection-demo` - sends image bytes plus a model path to the shim, which resolves and runs either a lightweight `.json` detector or a real `.onnx` detector on the host side and returns structured boxes.

All examples run as ordinary `scratch` OCI images. The guest only sees the `webgpu` ABI; the shim owns the native GPU stack and brokers requests on the host side.

Guest code should use the `webgpu_guest` helper crate in this workspace rather than declaring the raw `extern "C"` imports manually. That crate hides the low-level import block and provides safe helpers such as `describe_runtime()`, `execute(...)`, `detect(...)`, `bytes_from_f32_slice(...)`, and the typed `ComputeDispatch` and `ModelDetect` builders.

## Build

Build and install the shim on Linux:

```terminal
make build-webgpu
sudo make install-webgpu
```

Build all Wasm apps:

```terminal
make build-examples
```

Build all OCI tar artifacts:

```terminal
make build-examples-oci
```

The local tar artifacts are standard OCI/rootfs container images exported through Docker. The `.wasm` entrypoint lives inside the image filesystem and the shim runs it from the mounted container rootfs; these examples are not packaged through per-example Cargo `build.rs` scripts.

Build all example container images:

```terminal
make docker-build-examples
```

Push all images to the default `keniack` repository:

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
target/wasm32-wasip1/debug/yolo-detection-demo.wasm
target/wasm32-wasip1/debug/yolo-detection-demo-img.tar
```

The Docker targets publish:

```terminal
keniack/webgpu-demo:latest
keniack/image-classification-demo:latest
keniack/yolo-detection-demo:latest
```

## Execute

Import the local OCI images built by `make build-examples-oci`:

```terminal
sudo ctr images import --all-platforms target/wasm32-wasip1/debug/webgpu-demo-img.tar
sudo ctr images import --all-platforms target/wasm32-wasip1/debug/image-classification-demo-img.tar
sudo ctr images import --all-platforms target/wasm32-wasip1/debug/yolo-detection-demo-img.tar
```

Run `webgpu-demo`:

Use the explicit Wasm path form with `ctr run`. It matches the argv the guest expects and avoids accidental entrypoint overrides.

Guest argv:

```text
["/webgpu-demo.wasm", "dispatch", "16"]
```

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

The runtime output should report a real host adapter plus a successful dispatch. `dispatch.metadata` is intentionally compact: the shim strips the echoed WGSL source and replaces it with summary fields such as `shader_source_bytes` and `params_u32_len`.

Run `image-classification-demo`:

This example does not use `dispatch 16`. Its guest argv is:

```text
["/image-classification-demo.wasm", "<image-path>", "[model-path]"]
```

Use the image defaults:

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

Or pass the Wasm path plus explicit asset paths:

```terminal
sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  docker.io/keniack/image-classification-demo:local \
  image-classification-demo \
  /image-classification-demo.wasm /images/red-apple.ppm /models/resnet50-demo.json
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

Run `yolo-detection-demo`:

This example also does not use `dispatch 16`. Its guest argv is:

```text
["/yolo-detection-demo.wasm", "<image-path>", "[model-path]"]
```

Use the image defaults:

```terminal
sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  docker.io/keniack/yolo-detection-demo:local \
  yolo-detection-demo
```

The guest prefers `/models/yolov8l.onnx` when the image contains it. The repo image falls back to the bundled JSON detector so the smoke test stays small.

Pass the Wasm path plus explicit asset paths for the bundled JSON detector:

```terminal
sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  docker.io/keniack/yolo-detection-demo:local \
  yolo-detection-demo \
  /yolo-detection-demo.wasm /images/golden-retriever.ppm /models/yolo-detection-demo.json
```

Or use an ONNX model after adding it to the image:

```terminal
sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  docker.io/keniack/yolo-detection-demo:local \
  yolo-detection-demo \
  /yolo-detection-demo.wasm /images/golden-retriever.ppm /models/yolov8l.onnx
```

This example accepts both a lightweight `.json` demo model and a host-run `.onnx` model. The image can bundle `/models/yolov8l.onnx`, and when it is present the guest defaults to that path. The repo still keeps the tiny JSON detector as a fallback so the shipped smoke-test image stays small. The host running the shim needs `python3`, `onnxruntime`, `numpy`, and `Pillow` installed for `.onnx`.

For more detail, see:

- [examples/webgpu-demo/README.md](/Users/kenia/workspace/neuro-wasm/examples/webgpu-demo/README.md:1)
- [examples/image-classification-demo/README.md](/Users/kenia/workspace/neuro-wasm/examples/image-classification-demo/README.md:1)
- [examples/yolo-detection-demo/README.md](/Users/kenia/workspace/neuro-wasm/examples/yolo-detection-demo/README.md:1)
