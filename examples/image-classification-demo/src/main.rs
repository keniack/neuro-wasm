use std::env;
use std::fs;
use std::process;

use serde::Deserialize;
use webgpu_guest::{
    ComputeDispatch, DispatchResponse, ResultEncoding, RuntimeDescription, bytes_from_f32_slice,
    describe_runtime, execute,
};

const DEFAULT_MODEL_PATH: &str = "/models/resnet50-demo.json";
const CLASSIFIER_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(64)
fn matmul_logits(@builtin(global_invocation_id) gid: vec3<u32>) {
  let class_index = gid.x;
  let stride = params.x;
  let class_count = params.y;
  if (class_index >= class_count) {
    return;
  }

  let base = class_index * stride;
  var total: f32 = 0.0;
  for (var feature_index: u32 = 0u; feature_index < stride; feature_index = feature_index + 1u) {
    total = total + (weights[base + feature_index] * features[feature_index]);
  }

  logits[class_index] = total;
}
"#;

#[derive(Debug, Deserialize)]
struct DemoModel {
    name: String,
    #[serde(default = "default_task")]
    task: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    input_dim: usize,
    labels: Vec<DemoLabel>,
    weights: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct DemoLabel {
    id: String,
    label: String,
}

#[derive(Debug)]
struct ClassificationResponse {
    model: String,
    task: String,
    image_width: u32,
    image_height: u32,
    dispatch: DispatchResponse,
    top_k: Vec<Prediction>,
}

#[derive(Debug)]
struct Prediction {
    id: String,
    label: String,
    score: f32,
}

fn main() {
    let mut args = env::args().skip(1);
    let image_path = match args.next() {
        Some(path) => path,
        None => {
            eprintln!("usage: image-classification-demo <image-path> [model-path]");
            process::exit(1);
        }
    };
    let model_path = args
        .next()
        .unwrap_or_else(|| DEFAULT_MODEL_PATH.to_string());

    let model_bytes = fs::read(&model_path).unwrap_or_else(|err| {
        eprintln!("failed to read model at {model_path}: {err}");
        process::exit(2);
    });
    let image_bytes = fs::read(&image_path).unwrap_or_else(|err| {
        eprintln!("failed to read image at {image_path}: {err}");
        process::exit(2);
    });
    let model = serde_json::from_slice::<DemoModel>(&model_bytes).unwrap_or_else(|err| {
        eprintln!("failed to parse model at {model_path}: {err}");
        process::exit(2);
    });
    validate_model(&model, &model_path);
    let runtime = describe_runtime().unwrap_or_else(|err| {
        eprintln!("failed to query webgpu runtime: {err}");
        process::exit(3);
    });

    let response = classify_via_webgpu(&model, &image_bytes).unwrap_or_else(|err| {
        eprintln!("classification failed: {err}");
        process::exit(4);
    });

    println!("input.image={image_path}");
    println!("input.model={model_path}");
    print_runtime(&runtime);
    println!("runtime.task={}", response.task);
    println!("runtime.backend={}", response.dispatch.backend);
    println!("runtime.adapter_name={}", response.dispatch.adapter_name);
    println!(
        "image.dimensions={}x{}",
        response.image_width, response.image_height
    );
    println!("dispatch.kind={}", response.dispatch.kind);
    println!("dispatch.entrypoint={}", response.dispatch.entrypoint);
    println!(
        "dispatch.workgroups={}x{}x{}",
        response.dispatch.workgroups[0],
        response.dispatch.workgroups[1],
        response.dispatch.workgroups[2]
    );
    println!("dispatch.invocations={}", response.dispatch.invocations);
    println!("dispatch.checksum={}", response.dispatch.checksum);
    println!("dispatch.input_a_len={}", response.dispatch.input_a_len);
    println!("dispatch.input_b_len={}", response.dispatch.input_b_len);
    println!("dispatch.output_bytes={}", response.dispatch.output_bytes);
    println!(
        "dispatch.result_encoding={}",
        response.dispatch.result_encoding
    );
    println!("dispatch.output_words={:?}", response.dispatch.output_words);
    println!("dispatch.output_f32={:?}", response.dispatch.output_f32);
    println!("dispatch.metadata={}", response.dispatch.metadata);

    if let Some(best) = response.top_k.first() {
        println!("prediction.top1.id={}", best.id);
        println!("prediction.top1.label={}", best.label);
        println!("prediction.top1.score={:.4}", best.score);
    }

    for (index, prediction) in response.top_k.iter().enumerate() {
        println!("prediction.rank{}.id={}", index + 1, prediction.id);
        println!("prediction.rank{}.label={}", index + 1, prediction.label);
        println!("prediction.rank{}.score={:.4}", index + 1, prediction.score);
    }

    println!("prediction.model={}", response.model);
}

fn classify_via_webgpu(
    model: &DemoModel,
    image_bytes: &[u8],
) -> Result<ClassificationResponse, String> {
    let (features, image_width, image_height) = extract_image_features(image_bytes)?;
    let weight_bytes = bytes_from_f32_slice(&flatten_weights(model));
    let feature_bytes = bytes_from_f32_slice(&features_with_bias(&features));
    let stride =
        u32::try_from(model.input_dim + 1).map_err(|_| "model input dimension is too large")?;
    let class_count =
        u32::try_from(model.labels.len()).map_err(|_| "model class count is too large")?;

    let request = ComputeDispatch::new("matmul_logits", CLASSIFIER_SHADER, model.labels.len())
        .workgroups([div_ceil_u32(class_count, 64), 1, 1])
        .params_u32([stride, class_count, 0, 0])
        .result_encoding(ResultEncoding::F32)
        .metadata("task", model.task.clone())
        .metadata("model_name", model.name.clone())
        .metadata("input_type", "image-features");

    let dispatch: DispatchResponse = execute(&request, &weight_bytes, &feature_bytes)
        .map_err(|err| format!("dispatch failed: {err}"))?;
    let logits = dispatch
        .output_f32
        .clone()
        .ok_or_else(|| "dispatch response did not include f32 output".to_string())?;
    let top_k = decode_predictions(model, &logits);

    Ok(ClassificationResponse {
        model: model.name.clone(),
        task: model.task.clone(),
        image_width,
        image_height,
        dispatch,
        top_k,
    })
}

fn extract_image_features(image_bytes: &[u8]) -> Result<(Vec<f32>, u32, u32), String> {
    let image = image::load_from_memory(image_bytes)
        .map_err(|err| format!("failed to decode input image: {err}"))?;
    let rgb = image.to_rgb8();
    let pixel_count = (u64::from(rgb.width()) * u64::from(rgb.height())).max(1) as f32;

    let mut total_r = 0.0f32;
    let mut total_g = 0.0f32;
    let mut total_b = 0.0f32;
    let mut total_luma = 0.0f32;

    for pixel in rgb.pixels() {
        let r = f32::from(pixel[0]) / 255.0;
        let g = f32::from(pixel[1]) / 255.0;
        let b = f32::from(pixel[2]) / 255.0;
        total_r += r;
        total_g += g;
        total_b += b;
        total_luma += (0.299 * r) + (0.587 * g) + (0.114 * b);
    }

    Ok((
        vec![
            total_r / pixel_count,
            total_g / pixel_count,
            total_b / pixel_count,
            total_luma / pixel_count,
        ],
        rgb.width(),
        rgb.height(),
    ))
}

fn decode_predictions(model: &DemoModel, logits: &[f32]) -> Vec<Prediction> {
    let probabilities = softmax(logits);
    let mut predictions = model
        .labels
        .iter()
        .zip(probabilities.iter().copied())
        .map(|(label, score)| Prediction {
            id: label.id.clone(),
            label: label.label.clone(),
            score,
        })
        .collect::<Vec<_>>();

    predictions.sort_by(|left, right| right.score.total_cmp(&left.score));
    predictions.truncate(model.top_k.max(1).min(predictions.len()));
    predictions
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max_value = values.iter().copied().reduce(f32::max).unwrap_or(0.0);
    let exponentials = values
        .iter()
        .map(|value| (value - max_value).exp())
        .collect::<Vec<_>>();
    let total = exponentials.iter().copied().sum::<f32>().max(f32::EPSILON);

    exponentials
        .into_iter()
        .map(|value| value / total)
        .collect()
}

fn flatten_weights(model: &DemoModel) -> Vec<f32> {
    model
        .weights
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect()
}

fn features_with_bias(features: &[f32]) -> Vec<f32> {
    let mut values = features.to_vec();
    values.push(1.0);
    values
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

fn validate_model(model: &DemoModel, model_path: &str) {
    if model.labels.is_empty() {
        eprintln!("model at {model_path} has no labels");
        process::exit(2);
    }
    if model.weights.len() != model.labels.len() {
        eprintln!(
            "model at {model_path} has {} labels but {} weight rows",
            model.labels.len(),
            model.weights.len()
        );
        process::exit(2);
    }

    let expected_width = model.input_dim + 1;
    for (index, row) in model.weights.iter().enumerate() {
        if row.len() != expected_width {
            eprintln!(
                "model at {model_path} has invalid weight row {index}: expected {expected_width} entries, got {}",
                row.len()
            );
            process::exit(2);
        }
    }
}

fn default_task() -> String {
    "image-classification".to_string()
}

fn default_top_k() -> usize {
    3
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}
