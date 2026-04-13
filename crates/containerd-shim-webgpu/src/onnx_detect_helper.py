#!/usr/bin/env python3
import argparse
import json
import sys
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run host-side onnx detection")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--provider")
    return parser.parse_args()


def load_runtime():
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
    except Exception as exc:
        sys.stderr.write(
            "onnx detection requires the host to have `onnxruntime`, `numpy`, and `Pillow` "
            f"installed: {exc}\n"
        )
        raise SystemExit(1)
    return np, ort, Image


def select_providers(ort, requested: Optional[str]) -> List[str]:
    available = ort.get_available_providers()
    if requested:
        if requested not in available:
            sys.stderr.write(
                f"requested onnxruntime provider {requested!r} is unavailable. "
                f"Available providers: {available}\n"
            )
            raise SystemExit(1)
        return [requested]

    # Prefer hardware-backed EPs first, then fall back to CPU. This keeps the guest API
    # stable while letting the host decide what acceleration is actually present.
    priority = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "MIGraphXExecutionProvider",
        "OpenVINOExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "QNNExecutionProvider",
        "CPUExecutionProvider",
    ]
    selected = [provider for provider in priority if provider in available]
    return selected or available


def preprocess_image(Image, np, image_path: str, height: int, width: int):
    image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size
    resized = image.resize((width, height))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    return chw[np.newaxis, ...], orig_width, orig_height


def normalize_output(np, output):
    values = np.asarray(output)
    if values.ndim == 3:
        values = values[0]
        if values.shape[0] < values.shape[1]:
            values = values.transpose()
    elif values.ndim != 2:
        raise ValueError(f"unexpected ONNX output shape: {values.shape!r}")
    return values


def decode_predictions(np, predictions, score_threshold: float):
    if predictions.shape[1] < 5:
        raise ValueError(f"unexpected detection tensor width: {predictions.shape[1]}")

    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = scores >= score_threshold
    return boxes[keep], class_ids[keep], scores[keep]


def xywh_to_xyxy(box):
    center_x, center_y, width, height = box
    half_w = width / 2.0
    half_h = height / 2.0
    return [
        center_x - half_w,
        center_y - half_h,
        center_x + half_w,
        center_y + half_h,
    ]


def iou(left, right) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = max(left_area + right_area - intersection, 1e-6)
    return intersection / union


def non_max_suppression(detections: List[dict], iou_threshold: float, max_detections: int) -> List[dict]:
    detections.sort(key=lambda item: item["score"], reverse=True)
    kept: List[dict] = []
    for candidate in detections:
        if any(iou(candidate["bbox_xyxy"], kept_item["bbox_xyxy"]) >= iou_threshold for kept_item in kept):
            continue
        kept.append(candidate)
        if len(kept) >= max_detections:
            break
    return kept


def scale_box(box, input_width: int, input_height: int, output_width: int, output_height: int):
    x1, y1, x2, y2 = xywh_to_xyxy(box)
    x1 = max(0.0, min(float(output_width - 1), x1 / input_width * output_width))
    y1 = max(0.0, min(float(output_height - 1), y1 / input_height * output_height))
    x2 = max(0.0, min(float(output_width - 1), x2 / input_width * output_width))
    y2 = max(0.0, min(float(output_height - 1), y2 / input_height * output_height))
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def normalized_box(box, input_width: int, input_height: int):
    center_x, center_y, width, height = box
    return [
        float(center_x / input_width),
        float(center_y / input_height),
        float(width / input_width),
        float(height / input_height),
    ]


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        value = names.get(class_id)
        if value is None:
            value = names.get(str(class_id))
        return str(value if value is not None else class_id)
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def main() -> int:
    args = parse_args()
    np, ort, Image = load_runtime()
    providers = select_providers(ort, args.provider)

    try:
        session = ort.InferenceSession(args.model, providers=providers)
        input_meta = session.get_inputs()[0]
        input_shape = list(input_meta.shape)
        input_height = int(input_shape[2] if len(input_shape) > 2 and input_shape[2] else 640)
        input_width = int(input_shape[3] if len(input_shape) > 3 and input_shape[3] else 640)

        image, original_width, original_height = preprocess_image(
            Image, np, args.image, input_height, input_width
        )
        outputs = session.run(None, {input_meta.name: image})
        predictions = normalize_output(np, outputs[0])
        boxes, class_ids, scores = decode_predictions(np, predictions, args.score_threshold)

        metadata_map = session.get_modelmeta().custom_metadata_map or {}
        names_raw = metadata_map.get("names")
        names = {}
        if names_raw:
            try:
                names = json.loads(names_raw)
            except Exception:
                names = {}

        detections = []
        for index, (box, class_id, score) in enumerate(zip(boxes, class_ids, scores)):
            detections.append(
                {
                    "id": f"class_{int(class_id)}_{index}",
                    "label": class_name(names, int(class_id)),
                    "score": float(score),
                    "bbox_xywh_norm": normalized_box(box, input_width, input_height),
                    "bbox_xyxy": scale_box(
                        box, input_width, input_height, original_width, original_height
                    ),
                }
            )

        detections = non_max_suppression(detections, args.iou_threshold, args.max_detections)
        payload = {
            "detections": detections,
            "metadata": {
                "provider": session.get_providers()[0] if session.get_providers() else "unknown",
                "input_width": input_width,
                "input_height": input_height,
                "original_width": original_width,
                "original_height": original_height,
                "score_threshold": args.score_threshold,
                "iou_threshold": args.iou_threshold,
                "max_detections": args.max_detections,
                "runner_source": "shim",
            },
        }
        sys.stdout.write(json.dumps(payload))
        return 0
    except Exception as exc:
        sys.stderr.write(f"onnx detection failed: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
