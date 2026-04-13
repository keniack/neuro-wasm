use std::env;
use std::fs;
use std::path::Path;
use std::process;

use webgpu_guest::{DetectionResponse, ModelDetect, RuntimeDescription, describe_runtime, detect};

const DEFAULT_ONNX_MODEL_PATH: &str = "/models/yolov8l.onnx";
const FALLBACK_JSON_MODEL_PATH: &str = "/models/yolo-detection-demo.json";

fn main() {
    let mut args = env::args().skip(1);
    let image_path = match args.next() {
        Some(path) => path,
        None => {
            eprintln!("usage: yolo-detection-demo <image-path> [model-path]");
            process::exit(1);
        }
    };
    let model_path = args.next().unwrap_or_else(default_model_path);

    let image_bytes = fs::read(&image_path).unwrap_or_else(|err| {
        eprintln!("failed to read image at {image_path}: {err}");
        process::exit(2);
    });
    let runtime = describe_runtime().unwrap_or_else(|err| {
        eprintln!("failed to query webgpu runtime: {err}");
        process::exit(3);
    });

    // The guest only provides the image bytes and an in-container model path.
    // The shim resolves the path inside the bundle rootfs, loads the model on the host,
    // runs detection there, and returns structured results to the guest.
    let request = ModelDetect::new(model_path.clone())
        .task("object-detection")
        .score_threshold(0.25)
        .iou_threshold(0.45)
        .max_detections(20)
        .metadata("source", "examples/yolo-detection-demo");
    let response = detect(&request, &image_bytes).unwrap_or_else(|err| {
        eprintln!("detection failed: {err}");
        process::exit(4);
    });

    println!("input.image={image_path}");
    println!("input.model={model_path}");
    print_runtime(&runtime);
    print_detection_response(&response);
}

fn default_model_path() -> String {
    // Prefer a real ONNX checkpoint when the image contains one, but keep the tiny
    // bundled JSON model as a fallback so the repo still ships a runnable smoke test.
    if Path::new(DEFAULT_ONNX_MODEL_PATH).exists() {
        DEFAULT_ONNX_MODEL_PATH.to_string()
    } else {
        FALLBACK_JSON_MODEL_PATH.to_string()
    }
}

fn print_runtime(runtime: &RuntimeDescription) {
    println!("webgpu.enabled={}", runtime.enabled);
    println!("webgpu.required={}", runtime.required);
    println!("webgpu.backend={}", runtime.backend);
    println!("webgpu.adapter_name={}", runtime.adapter_name);
    println!("webgpu.device_available={}", runtime.device_available);
    println!("webgpu.runtime_ready={}", runtime.runtime_ready);
    println!(
        "webgpu.runtime_error={}",
        runtime.runtime_error.as_deref().unwrap_or("none")
    );
    println!("webgpu.max_buffer_size={}", runtime.max_buffer_size);
    println!("webgpu.max_bind_groups={}", runtime.max_bind_groups);
    println!("webgpu.fallback_adapter={}", runtime.force_fallback_adapter);
}

fn print_detection_response(response: &DetectionResponse) {
    println!("detection.kind={}", response.kind);
    println!("detection.task={}", response.task);
    println!("detection.model_path={}", response.model_path);
    println!("detection.model_format={}", response.model_format);
    println!("detection.runner={}", response.runner);
    println!("detection.image_bytes={}", response.image_bytes);
    println!("detection.count={}", response.detection_count);
    println!("detection.metadata={}", response.metadata);

    for (index, detection) in response.detections.iter().enumerate() {
        println!("detection.rank{}.id={}", index + 1, detection.id);
        println!("detection.rank{}.label={}", index + 1, detection.label);
        println!("detection.rank{}.score={:.4}", index + 1, detection.score);
        println!(
            "detection.rank{}.bbox_xywh_norm={:?}",
            index + 1,
            detection.bbox_xywh_norm
        );
        println!(
            "detection.rank{}.bbox_xyxy={:?}",
            index + 1,
            detection.bbox_xyxy
        );
    }
}
