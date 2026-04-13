# yolo-detection-demo

`yolo-detection-demo` is a shim-owned object-detection demo for the experimental `containerd-shim-webgpu-v1` shim.

The workload:

- takes the image path as the first CLI argument
- calls `webgpu.describe_runtime` to query the effective shim-side runtime
- reads the image bytes inside the guest
- calls the workspace `webgpu_guest::detect(...)` helper
- sends the image bytes to the shim
- lets the shim load the configured host model and run detection outside the container
- receives structured detections back from the shim

This is a separate demo from `image-classification-demo` because the shim owns the full model execution path here rather than exposing only a generic compute kernel.

## Model

The demo uses ONNX models executed through a host-side ONNX Runtime helper. The host running `containerd-shim-webgpu-v1` must have:

- `python3`
- `onnxruntime`
- `numpy`
- `Pillow`

The shim chooses the best available ONNX Runtime execution provider automatically and falls back to CPU. You can pin a specific provider through `ModelDetect::provider(...)` in the guest SDK.

## Model Selection

The container image does not bundle a model. Set `WEBGPU_MODEL` to the full path of the ONNX model on the host. The shim loads that file directly and the guest never sees the host path as an input.

For example:

```text
/host/models/model.onnx
```

`WEBGPU_MODEL` is host-only and is never forwarded into the guest environment.

## Bundled Assets

- shared sample images from `examples/image-classification-demo/images`

The bundled image defaults are:

- `/images/golden-retriever.ppm`
- `/images/ocean.ppm`
- `/images/red-apple.ppm`

## Build The Wasm Module

```terminal
cargo build \
  -p yolo-detection-demo \
  --target wasm32-wasip1
```

This produces:

```terminal
target/wasm32-wasip1/debug/yolo-detection-demo.wasm
```

## Build An OCI Tar

```terminal
make build-examples-oci
```

This exports a standard OCI/rootfs container image tar to:

```terminal
target/wasm32-wasip1/debug/yolo-detection-demo-img.tar
```

## Build And Push A Registry Image

Build the example image:

```terminal
make docker-build-yolo-detection-demo
```

Push it to the default `keniack` repository:

```terminal
make docker-push-yolo-detection-demo
```

Override the repository or tag when needed:

```terminal
make docker-push-yolo-detection-demo IMAGE_REPO=keniack IMAGE_TAG=v0.1.0
```

## Run Through The Shim

This example does not use `dispatch 16`. The guest argv shape is:

```text
["/yolo-detection-demo.wasm", "<image-path>"]
```

Run with `WEBGPU_MODEL` pointing to the model file on the host:

```terminal
sudo ctr images pull docker.io/keniack/yolo-detection-demo:latest

sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  --env WEBGPU_MODEL=/host/models/model.onnx \
  docker.io/keniack/yolo-detection-demo:latest \
  yolo-detection-demo
```

Pass an explicit image path to override the default image:

```terminal
sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  --env WEBGPU_MODEL=/host/models/model.onnx \
  docker.io/keniack/yolo-detection-demo:latest \
  yolo-detection-demo \
  /yolo-detection-demo.wasm /images/golden-retriever.ppm
```

On success, the output should include:

- `webgpu.runtime_ready=true`
- the real host adapter name
- ranked detections with `bbox_xywh_norm` and `bbox_xyxy`
- `detection.runner=onnxruntime.host`

This image stays `scratch`. The WebGPU shim keeps Vulkan on the host side and brokers WebGPU requests from the guest; host-only settings such as `WEBGPU_DEVICE_PATH` and `WEBGPU_MODEL` are not forwarded into the guest Wasm environment.
