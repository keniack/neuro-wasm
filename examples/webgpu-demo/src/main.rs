use std::env;
use std::process;

use serde::{Deserialize, Serialize};

const OUTPUT_BUFFER_SIZE: usize = 16 * 1024;
const VECTOR_ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_buffer: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(64)
fn add_vectors(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.x) {
    return;
  }

  output_buffer[index] = input_a[index] + input_b[index];
}
"#;

#[derive(Debug, Deserialize)]
struct RuntimeDescription {
    enabled: bool,
    backend: String,
    adapter_name: String,
    device_path: Option<String>,
    device_available: bool,
    runtime_ready: bool,
    runtime_error: Option<String>,
    max_buffer_size: u64,
    max_bind_groups: u32,
    force_fallback_adapter: bool,
    required: bool,
}

#[derive(Debug, Serialize)]
struct ExecuteRequest<'a> {
    kind: &'a str,
    entrypoint: &'a str,
    workgroups: Option<[u32; 3]>,
    output_words: usize,
    metadata: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct DispatchResponse {
    kind: String,
    entrypoint: String,
    backend: String,
    adapter_name: String,
    device_path: Option<String>,
    workgroups: [u32; 3],
    invocations: u32,
    checksum: u64,
    input_a_len: usize,
    input_b_len: usize,
    output_bytes: usize,
    result_encoding: String,
    output_words: Vec<u32>,
    output_f32: Option<Vec<f32>>,
    metadata: serde_json::Value,
}

#[link(wasm_import_module = "webgpu")]
unsafe extern "C" {
    #[link_name = "describe_runtime"]
    fn webgpu_describe_runtime(output_ptr: *mut u8, output_cap: i32) -> i32;

    #[link_name = "execute"]
    fn webgpu_execute(
        request_ptr: *const u8,
        request_len: i32,
        input_a_ptr: *const u8,
        input_a_len: i32,
        input_b_ptr: *const u8,
        input_b_len: i32,
        output_ptr: *mut u8,
        output_cap: i32,
    ) -> i32;
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("summary");

    let runtime = describe_runtime().unwrap_or_else(|err| {
        eprintln!("failed to query webgpu runtime: {err}");
        process::exit(2);
    });

    match command {
        "summary" => print_runtime(&runtime),
        "dispatch" => {
            let element_count = parse_dim(args.get(2), 8).max(1) as usize;
            let input_a = (0..element_count)
                .map(|index| (index + 1) as f32)
                .collect::<Vec<_>>();
            let input_b = (0..element_count)
                .map(|index| ((index + 1) * 10) as f32)
                .collect::<Vec<_>>();
            let request = ExecuteRequest {
                kind: "compute.dispatch",
                entrypoint: "add_vectors",
                workgroups: Some([div_ceil_u32(element_count as u32, 64), 1, 1]),
                output_words: element_count,
                metadata: serde_json::json!({
                    "shader_source": VECTOR_ADD_SHADER,
                    "params_u32": [element_count as u32, 0, 0, 0],
                    "result_encoding": "f32",
                    "label": "vector-add",
                    "source": "examples/webgpu-demo"
                }),
            };
            let response = execute_request(
                &request,
                &encode_f32_slice(&input_a),
                &encode_f32_slice(&input_b),
            )
            .unwrap_or_else(|err| {
                eprintln!("dispatch failed: {err}");
                process::exit(3);
            });
            let dispatch: DispatchResponse =
                serde_json::from_slice(&response).unwrap_or_else(|err| {
                    eprintln!("invalid dispatch response: {err}");
                    process::exit(4);
                });

            print_runtime(&runtime);
            println!("dispatch.kind={}", dispatch.kind);
            println!("dispatch.entrypoint={}", dispatch.entrypoint);
            println!("dispatch.backend={}", dispatch.backend);
            println!("dispatch.adapter_name={}", dispatch.adapter_name);
            println!(
                "dispatch.device_path={}",
                dispatch.device_path.as_deref().unwrap_or("unset")
            );
            println!(
                "dispatch.workgroups={}x{}x{}",
                dispatch.workgroups[0], dispatch.workgroups[1], dispatch.workgroups[2]
            );
            println!("dispatch.invocations={}", dispatch.invocations);
            println!("dispatch.checksum={}", dispatch.checksum);
            println!("dispatch.input_a_len={}", dispatch.input_a_len);
            println!("dispatch.input_b_len={}", dispatch.input_b_len);
            println!("dispatch.output_bytes={}", dispatch.output_bytes);
            println!("dispatch.result_encoding={}", dispatch.result_encoding);
            println!("dispatch.output_words={:?}", dispatch.output_words);
            println!("dispatch.output_f32={:?}", dispatch.output_f32);
            println!("dispatch.metadata={}", dispatch.metadata);
        }
        "validate" => {
            print_runtime(&runtime);
            println!("validation=ok");
        }
        other => {
            eprintln!("unknown command: {other}");
            eprintln!("usage: webgpu-demo [summary|dispatch [element-count]|validate]");
            process::exit(1);
        }
    }
}

fn describe_runtime() -> Result<RuntimeDescription, String> {
    let mut output = vec![0u8; OUTPUT_BUFFER_SIZE];
    let written = unsafe {
        webgpu_describe_runtime(
            output.as_mut_ptr(),
            i32::try_from(output.len()).map_err(|_| "output buffer too large")?,
        )
    };

    if written < 0 {
        return Err("webgpu.describe_runtime returned an error".to_string());
    }

    let written = usize::try_from(written).map_err(|_| "invalid runtime description size")?;
    serde_json::from_slice::<RuntimeDescription>(&output[..written])
        .map_err(|err| format!("invalid runtime description json: {err}"))
}

fn execute_request(
    request: &ExecuteRequest<'_>,
    input_a: &[u8],
    input_b: &[u8],
) -> Result<Vec<u8>, String> {
    let request_bytes =
        serde_json::to_vec(request).map_err(|err| format!("invalid request json: {err}"))?;
    let mut output = vec![0u8; OUTPUT_BUFFER_SIZE];

    let written = unsafe {
        webgpu_execute(
            request_bytes.as_ptr(),
            i32::try_from(request_bytes.len()).map_err(|_| "request too large")?,
            input_a.as_ptr(),
            i32::try_from(input_a.len()).map_err(|_| "input_a too large")?,
            input_b.as_ptr(),
            i32::try_from(input_b.len()).map_err(|_| "input_b too large")?,
            output.as_mut_ptr(),
            i32::try_from(output.len()).map_err(|_| "output buffer too large")?,
        )
    };

    if written < 0 {
        return Err("webgpu.execute returned an error".to_string());
    }

    let written = usize::try_from(written).map_err(|_| "invalid execute response size")?;
    output.truncate(written);
    Ok(output)
}

fn print_runtime(runtime: &RuntimeDescription) {
    println!("webgpu.enabled={}", runtime.enabled);
    println!("webgpu.required={}", runtime.required);
    println!("webgpu.backend={}", runtime.backend);
    println!("webgpu.adapter_name={}", runtime.adapter_name);
    println!(
        "webgpu.device_path={}",
        runtime.device_path.as_deref().unwrap_or("unset")
    );
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

fn encode_f32_slice(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn parse_dim(value: Option<&String>, default: u32) -> u32 {
    value
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}
