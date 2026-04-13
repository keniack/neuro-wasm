use std::env;
use std::process;

use webgpu_guest::{
    ComputeDispatch, DispatchResponse, ResultEncoding, RuntimeDescription, bytes_from_f32_slice,
    describe_runtime, execute,
};

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
            let request = ComputeDispatch::new("add_vectors", VECTOR_ADD_SHADER, element_count)
                .workgroups([div_ceil_u32(element_count as u32, 64), 1, 1])
                .params_u32([element_count as u32, 0, 0, 0])
                .result_encoding(ResultEncoding::F32)
                .metadata("label", "vector-add")
                .metadata("source", "examples/webgpu-demo");
            let dispatch: DispatchResponse = execute(
                &request,
                &bytes_from_f32_slice(&input_a),
                &bytes_from_f32_slice(&input_b),
            )
            .unwrap_or_else(|err| {
                eprintln!("dispatch failed: {err}");
                process::exit(3);
            });

            print_runtime(&runtime);
            println!("dispatch.kind={}", dispatch.kind);
            println!("dispatch.entrypoint={}", dispatch.entrypoint);
            println!("dispatch.backend={}", dispatch.backend);
            println!("dispatch.adapter_name={}", dispatch.adapter_name);
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

fn parse_dim(value: Option<&String>, default: u32) -> u32 {
    value
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}
