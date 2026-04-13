# yolo-detection-demo

`yolo-detection-demo` is a shim-owned object-detection demo for the experimental `containerd-shim-webgpu-v1` shim.

The workload:

- takes the image path as the first CLI argument
- optionally takes a model path as the second CLI argument
- calls `webgpu.describe_runtime` to query the effective shim-side runtime
- reads the image bytes inside the guest
- calls the workspace `webgpu_guest::detect(...)` helper
- sends the image bytes plus the in-container model path to the shim
- lets the shim resolve the model inside the container rootfs and execute detection on the host
- receives structured detections back from the shim

This is a separate demo from `image-classification-demo` because the shim owns the full model execution path here rather than exposing only a generic compute kernel.

## Model Format

This demo accepts two model formats:

- `.json` - the lightweight demo detector bundled in this repo
- `.onnx` - a host-run ONNX detector such as `yolov8l.onnx`

The guest stays the same in both cases. Only the shim changes runner behavior based on the model extension.

`.onnx` execution happens in the shim broker on the host and requires:

- `python3`
- `onnxruntime`
- `numpy`
- `Pillow`

The shim can optionally pin a specific ONNX Runtime provider through
`ModelDetect::provider(...)`, but by default it will choose the best provider
available on the host and fall back to CPU when needed.

## Bundled Assets

- shared sample images from `examples/image-classification-demo/images`
- `models/yolo-detection-demo.json`

If you want to test a real ONNX model, you have two options:

**Bundle it in the image** — place it under:

```text
examples/yolo-detection-demo/models/yolov8l.onnx
```

before rebuilding, or build your own image that includes the model at a path such as `/models/yolov8l.onnx`.

**Keep it on the host** — set `WEBGPU_MODEL_DIR` to the directory on the host machine where models live. The shim will resolve the guest model path against that directory when the file is not found in the container rootfs. For example:

```terminal
sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  --env WEBGPU_MODEL_DIR=/host/models \
  docker.io/keniack/yolo-detection-demo:latest \
  yolo-detection-demo
```

With `WEBGPU_MODEL_DIR=/host/models` the shim looks for the model at `/host/models/examples/yolo-detection-demo/models/model.onnx` (or whichever path the guest requested). `WEBGPU_MODEL_DIR` is host-only and is not forwarded into the guest environment.

The demo guest prefers `/models/yolov8l.onnx` when the image contains it. If not, it falls back to the bundled JSON detector so the repo still ships a small runnable image.

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
["/yolo-detection-demo.wasm", "<image-path>", "[model-path]"]
```

Import the local OCI tar and run the pre-bundled demo image:

```terminal
sudo ctr images import --all-platforms target/wasm32-wasip1/debug/yolo-detection-demo-img.tar

sudo ctr images pull docker.io/keniack/yolo-detection-demo:latest

sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  docker.io/keniack/yolo-detection-demo:latest \
  yolo-detection-demo
```

Or pass the Wasm path plus explicit asset paths for the bundled JSON detector:

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

To run a real `.onnx` model bundled inside the image:

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

Or load the model from the host via `WEBGPU_MODEL_DIR` without rebuilding the image:

```terminal
sudo ctr run --rm \
  --runtime=io.containerd.webgpu.v1 \
  --env WEBGPU_ENABLED=1 \
  --env WEBGPU_REQUIRED=1 \
  --env WEBGPU_BACKEND=vulkan \
  --env WEBGPU_DEVICE_PATH=/dev/dri/renderD128 \
  --env WEBGPU_MODEL_DIR=/host/models \
  docker.io/keniack/yolo-detection-demo:latest \
  yolo-detection-demo \
  /yolo-detection-demo.wasm /images/golden-retriever.ppm examples/yolo-detection-demo/models/model.onnx
```

On success, the output should include:

- `webgpu.runtime_ready=true`
- the real host adapter name
- ranked detections with `bbox_xywh_norm` and `bbox_xyxy`
- `detection.runner=webgpu.json-model` for `.json`, or `detection.runner=onnxruntime.host` for `.onnx`

This image stays `scratch`. The WebGPU shim keeps Vulkan on the host side and brokers WebGPU requests from the guest; host-only settings such as `WEBGPU_DEVICE_PATH` are not forwarded into the guest Wasm environment.
