use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::host::{self, WebGpuExecutionRequest, WebGpuHostState};

const JSON_DETECTOR_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(64)
fn score_detections(@builtin(global_invocation_id) gid: vec3<u32>) {
  let detection_index = gid.x;
  let stride = params.x;
  let detection_count = params.y;
  if (detection_index >= detection_count) {
    return;
  }

  let base = detection_index * stride;
  var total: f32 = 0.0;
  for (var feature_index: u32 = 0u; feature_index < stride; feature_index = feature_index + 1u) {
    total = total + (weights[base + feature_index] * features[feature_index]);
  }

  scores[detection_index] = total;
}
"#;

const ONNX_HELPER_SCRIPT: &str = include_str!("onnx_detect_helper.py");
const ONNX_HELPER_NAME: &str = "runwasi_webgpu_onnx_detect.py";

static ONNX_HELPER_PATH: OnceLock<PathBuf> = OnceLock::new();

#[derive(Debug, Deserialize)]
struct ModelDetectMetadata {
    model_path: String,
    #[serde(default = "default_task")]
    task: String,
    #[serde(default = "default_score_threshold")]
    score_threshold: f32,
    #[serde(default = "default_iou_threshold")]
    iou_threshold: f32,
    #[serde(default = "default_max_detections")]
    max_detections: usize,
    #[serde(default)]
    provider: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DemoJsonModel {
    name: String,
    #[serde(default = "default_task")]
    task: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_score_threshold")]
    score_threshold: f32,
    input_dim: usize,
    labels: Vec<DemoLabel>,
    detections: Vec<DetectionPrototype>,
}

#[derive(Debug, Deserialize)]
struct DemoLabel {
    id: String,
    label: String,
}

#[derive(Debug, Deserialize)]
struct DetectionPrototype {
    id: String,
    label_id: String,
    bbox_xywh: [f32; 4],
    weights: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Detection {
    id: String,
    label: String,
    score: f32,
    bbox_xywh_norm: [f32; 4],
    bbox_xyxy: [u32; 4],
}

#[derive(Debug, Serialize)]
struct ModelDetectResponse {
    kind: &'static str,
    task: String,
    model_path: String,
    model_format: &'static str,
    runner: &'static str,
    image_bytes: usize,
    detection_count: usize,
    detections: Vec<Detection>,
    metadata: Value,
}

#[derive(Debug, Deserialize)]
struct OnnxDetectResponse {
    #[serde(default)]
    detections: Vec<Detection>,
    #[serde(default)]
    metadata: Value,
}

pub(crate) fn execute_model_detect(
    state: &WebGpuHostState,
    _request: &WebGpuExecutionRequest,
    image_bytes: &[u8],
) -> Result<Vec<u8>> {
    let metadata = parse_detect_metadata(&_request.metadata)?;
    let model_host_path = resolve_guest_path(state, &metadata.model_path)?;

    let response = match model_extension(&model_host_path) {
        "json" => {
            host::validate_runtime_access(state.config())
                .map_err(|err| anyhow::anyhow!("runtime access denied: {err:?}"))?;
            execute_json_detection(state, &metadata, &model_host_path, image_bytes)?
        }
        "onnx" => execute_onnx_detection(&metadata, &model_host_path, image_bytes)?,
        other => bail!("unsupported model format for model.detect: {other}"),
    };

    serde_json::to_vec(&response).context("serializing model.detect response")
}

fn parse_detect_metadata(metadata: &Value) -> Result<ModelDetectMetadata> {
    serde_json::from_value(metadata.clone()).context("parsing model.detect metadata")
}

fn resolve_guest_path(state: &WebGpuHostState, guest_path: &str) -> Result<PathBuf> {
    let raw_path = Path::new(guest_path);
    let mut relative = PathBuf::new();

    // The guest may pass an absolute in-container path like `/models/yolov8l.onnx`.
    // Strip the root and reject any path that tries to escape via `..`.
    for component in raw_path.components() {
        match component {
            Component::RootDir | Component::CurDir => {}
            Component::Normal(segment) => relative.push(segment),
            Component::ParentDir | Component::Prefix(_) => {
                bail!("model_path must not escape the model directory")
            }
        }
    }

    if relative.as_os_str().is_empty() {
        bail!("model_path must not be empty");
    }

    // Try the container rootfs first.
    if let Some(rootfs_dir) = state.rootfs_dir() {
        let resolved = rootfs_dir.join(&relative);
        if resolved.exists() {
            return Ok(resolved);
        }
    }

    // Fall back to the host model directory configured via WEBGPU_MODEL_DIR.
    if let Some(model_dir) = state.config().model_dir.as_deref() {
        let resolved = Path::new(model_dir).join(&relative);
        if resolved.exists() {
            return Ok(resolved);
        }
    }

    bail!(
        "model path {} does not exist in the container rootfs or WEBGPU_MODEL_DIR",
        guest_path
    )
}

fn execute_json_detection(
    state: &WebGpuHostState,
    request: &ModelDetectMetadata,
    model_path: &Path,
    image_bytes: &[u8],
) -> Result<ModelDetectResponse> {
    let model_bytes = fs::read(model_path)
        .with_context(|| format!("reading json detection model at {}", model_path.display()))?;
    let model: DemoJsonModel =
        serde_json::from_slice(&model_bytes).context("parsing json detection model")?;
    validate_json_model(&model, model_path)?;

    let runtime = host::ensure_runtime(state)?;
    let (features, image_width, image_height) = extract_image_features(image_bytes)?;
    let weight_bytes = bytes_from_f32_slice(&flatten_weights(&model));
    let feature_bytes = bytes_from_f32_slice(&features_with_bias(&features));
    let stride =
        u32::try_from(model.input_dim + 1).context("model input dimension is too large")?;
    let detection_count =
        u32::try_from(model.detections.len()).context("model detection count is too large")?;
    let output_size = u64::try_from(model.detections.len().saturating_mul(4))
        .context("output size is too large")?;

    // The shim-owned runner keeps the actual GPU work outside the guest while
    // still using the same WebGPU compute path under the hood for lightweight
    // demo models.
    let output_bytes = runtime.run_compute(
        JSON_DETECTOR_SHADER,
        "score_detections",
        [div_ceil_u32(detection_count, 64), 1, 1],
        &weight_bytes,
        &feature_bytes,
        &uniform_bytes(&[stride, detection_count, 0, 0]),
        output_size,
    )?;
    let logits = bytes_to_f32(&output_bytes)?;
    let detections = decode_json_detections(
        &model,
        &logits,
        image_width,
        image_height,
        request.iou_threshold,
    )?;

    Ok(ModelDetectResponse {
        kind: "model.detect",
        task: model.task.clone(),
        model_path: request.model_path.clone(),
        model_format: "json",
        runner: "webgpu.json-model",
        image_bytes: image_bytes.len(),
        detection_count: detections.len(),
        detections,
        metadata: json!({
            "model_name": model.name,
            "top_k": model.top_k,
            "score_threshold": model.score_threshold,
            "iou_threshold": request.iou_threshold,
            "image_width": image_width,
            "image_height": image_height,
            "runner_source": "shim"
        }),
    })
}

fn execute_onnx_detection(
    request: &ModelDetectMetadata,
    model_path: &Path,
    image_bytes: &[u8],
) -> Result<ModelDetectResponse> {
    let helper_path = helper_script_path()?;
    let temp_image = TempImage::write(image_bytes)?;
    let mut command = Command::new("python3");
    // Keep the heavier ONNX runtime out of the guest process. The guest only supplies
    // bytes and a model path; the shim owns model loading, preprocessing, and inference.
    command
        .arg(helper_path)
        .arg("--model")
        .arg(model_path)
        .arg("--image")
        .arg(&temp_image.path)
        .arg("--score-threshold")
        .arg(request.score_threshold.to_string())
        .arg("--iou-threshold")
        .arg(request.iou_threshold.to_string())
        .arg("--max-detections")
        .arg(request.max_detections.to_string());

    if let Some(provider) = request
        .provider
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        // Keep provider selection host-owned and optional. The guest can request one,
        // but the shim still decides whether the host runtime can actually honor it.
        command.arg("--provider").arg(provider);
    }

    let output = command
        .output()
        .context("starting host-side ONNX detection runner")?;

    if !output.status.success() {
        bail!(
            "onnx model runner failed: {}{}{}",
            String::from_utf8_lossy(&output.stderr),
            if output.stderr.is_empty() { "" } else { " " },
            String::from_utf8_lossy(&output.stdout)
        );
    }

    let runner: OnnxDetectResponse = serde_json::from_slice(&output.stdout)
        .context("parsing host-side onnx detection response")?;

    Ok(ModelDetectResponse {
        kind: "model.detect",
        task: request.task.clone(),
        model_path: request.model_path.clone(),
        model_format: "onnx",
        runner: "onnxruntime.host",
        image_bytes: image_bytes.len(),
        detection_count: runner.detections.len(),
        detections: runner.detections,
        metadata: runner.metadata,
    })
}

fn helper_script_path() -> Result<&'static Path> {
    if let Some(path) = ONNX_HELPER_PATH.get() {
        return Ok(path.as_path());
    }
    let dir = std::env::temp_dir().join("runwasi-webgpu");
    fs::create_dir_all(&dir)
        .with_context(|| format!("creating onnx helper directory at {}", dir.display()))?;
    let path = dir.join(ONNX_HELPER_NAME);
    fs::write(&path, ONNX_HELPER_SCRIPT)
        .with_context(|| format!("writing onnx helper script to {}", path.display()))?;
    Ok(ONNX_HELPER_PATH.get_or_init(|| path).as_path())
}

fn extract_image_features(image_bytes: &[u8]) -> Result<(Vec<f32>, u32, u32)> {
    let image = image::load_from_memory(image_bytes).context("decoding input image")?;
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

fn decode_json_detections(
    model: &DemoJsonModel,
    logits: &[f32],
    image_width: u32,
    image_height: u32,
    iou_threshold: f32,
) -> Result<Vec<Detection>> {
    if logits.len() != model.detections.len() {
        bail!(
            "json detection model produced {} logits for {} detections",
            logits.len(),
            model.detections.len()
        );
    }

    let labels = model
        .labels
        .iter()
        .map(|label| (label.id.as_str(), label.label.as_str()))
        .collect::<HashMap<_, _>>();

    let mut detections = model
        .detections
        .iter()
        .zip(logits.iter().copied())
        .filter_map(|(prototype, logit)| {
            let score = sigmoid(logit);
            if score < model.score_threshold {
                return None;
            }

            let label = labels.get(prototype.label_id.as_str())?;
            Some(Detection {
                id: prototype.id.clone(),
                label: (*label).to_string(),
                score,
                bbox_xywh_norm: prototype.bbox_xywh,
                bbox_xyxy: bbox_xywh_to_xyxy(prototype.bbox_xywh, image_width, image_height),
            })
        })
        .collect::<Vec<_>>();

    detections.sort_by(|left, right| right.score.total_cmp(&left.score));
    detections = non_max_suppression(detections, iou_threshold);
    detections.truncate(model.top_k.max(1).min(detections.len()));

    Ok(detections)
}

fn validate_json_model(model: &DemoJsonModel, model_path: &Path) -> Result<()> {
    if model.labels.is_empty() {
        bail!("json model at {} has no labels", model_path.display());
    }
    if model.detections.is_empty() {
        bail!("json model at {} has no detections", model_path.display());
    }

    let expected_width = model.input_dim + 1;
    let labels = model
        .labels
        .iter()
        .map(|label| label.id.as_str())
        .collect::<HashSet<_>>();

    for (index, detection) in model.detections.iter().enumerate() {
        if !labels.contains(detection.label_id.as_str()) {
            bail!(
                "json model at {} has detection {} referencing unknown label {}",
                model_path.display(),
                index,
                detection.label_id
            );
        }
        if detection.weights.len() != expected_width {
            bail!(
                "json model at {} has invalid weight row {}: expected {} entries, got {}",
                model_path.display(),
                index,
                expected_width,
                detection.weights.len()
            );
        }
    }

    Ok(())
}

fn flatten_weights(model: &DemoJsonModel) -> Vec<f32> {
    model
        .detections
        .iter()
        .flat_map(|detection| detection.weights.iter().copied())
        .collect()
}

fn features_with_bias(features: &[f32]) -> Vec<f32> {
    let mut values = features.to_vec();
    values.push(1.0);
    values
}

fn bytes_from_f32_slice(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>()
}

fn uniform_bytes(values: &[u32]) -> Vec<u8> {
    let raw = if values.is_empty() {
        vec![0; 16]
    } else {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let padded_len = raw.len().max(16).next_multiple_of(16);
    let mut padded = vec![0u8; padded_len];
    padded[..raw.len()].copy_from_slice(&raw);
    padded
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("GPU output buffer size is not divisible by 4");
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap_or([0; 4])))
        .collect())
}

fn bbox_xywh_to_xyxy(bbox_xywh: [f32; 4], image_width: u32, image_height: u32) -> [u32; 4] {
    let [center_x, center_y, width, height] = bbox_xywh;
    let min_x = (center_x - (width / 2.0)).clamp(0.0, 1.0);
    let min_y = (center_y - (height / 2.0)).clamp(0.0, 1.0);
    let max_x = (center_x + (width / 2.0)).clamp(0.0, 1.0);
    let max_y = (center_y + (height / 2.0)).clamp(0.0, 1.0);

    [
        scale_coord(min_x, image_width),
        scale_coord(min_y, image_height),
        scale_coord(max_x, image_width),
        scale_coord(max_y, image_height),
    ]
}

fn scale_coord(value: f32, extent: u32) -> u32 {
    let max_index = extent.saturating_sub(1);
    (value * (max_index as f32)).round() as u32
}

fn non_max_suppression(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    let mut kept = Vec::with_capacity(detections.len());

    while !detections.is_empty() {
        let candidate = detections.remove(0);
        detections.retain(|other| iou(candidate.bbox_xyxy, other.bbox_xyxy) < iou_threshold);
        kept.push(candidate);
    }

    kept
}

fn iou(left: [u32; 4], right: [u32; 4]) -> f32 {
    let inter_x1 = left[0].max(right[0]);
    let inter_y1 = left[1].max(right[1]);
    let inter_x2 = left[2].min(right[2]);
    let inter_y2 = left[3].min(right[3]);

    if inter_x2 <= inter_x1 || inter_y2 <= inter_y1 {
        return 0.0;
    }

    let inter_area = (inter_x2 - inter_x1) as f32 * (inter_y2 - inter_y1) as f32;
    let left_area = left[2].saturating_sub(left[0]) as f32 * left[3].saturating_sub(left[1]) as f32;
    let right_area =
        right[2].saturating_sub(right[0]) as f32 * right[3].saturating_sub(right[1]) as f32;
    let union_area = (left_area + right_area - inter_area).max(f32::EPSILON);

    inter_area / union_area
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

fn model_extension(path: &Path) -> &str {
    path.extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
}

fn default_task() -> String {
    "object-detection".to_string()
}

fn default_score_threshold() -> f32 {
    0.25
}

fn default_iou_threshold() -> f32 {
    0.45
}

fn default_max_detections() -> usize {
    20
}

fn default_top_k() -> usize {
    3
}

struct TempImage {
    path: PathBuf,
}

impl TempImage {
    fn write(bytes: &[u8]) -> Result<Self> {
        let dir = std::env::temp_dir().join("runwasi-webgpu-images");
        fs::create_dir_all(&dir)
            .with_context(|| format!("creating temporary image directory {}", dir.display()))?;
        let token = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = dir.join(format!("input-{token:x}{}", detect_image_extension(bytes)));
        fs::write(&path, bytes)
            .with_context(|| format!("writing temporary detection image to {}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for TempImage {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn detect_image_extension(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        ".png"
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        ".jpg"
    } else if bytes.starts_with(b"P6") || bytes.starts_with(b"P3") {
        ".ppm"
    } else {
        ".img"
    }
}
