use std::env;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use wasmedge_sdk::error::{CoreError, CoreExecutionError};
use wasmedge_sdk::{CallingFrame, ImportObject, ImportObjectBuilder, Instance, WasmValue};

use crate::middleware::WebGpuConfig;

const WEBGPU_IMPORT_MODULE: &str = "webgpu";
const DESCRIBE_RUNTIME_FN: &str = "describe_runtime";
const EXECUTE_FN: &str = "execute";

#[derive(Clone)]
pub(crate) struct WebGpuHostState {
    config: WebGpuConfig,
    broker_addr: Option<String>,
    // Reuse a single native device/queue across guest host calls inside the sandbox.
    runtime: Arc<OnceLock<Arc<WebGpuRuntime>>>,
}

struct WebGpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    backend: String,
    adapter_name: String,
    max_buffer_size: u64,
    max_bind_groups: u32,
}

#[derive(Debug, Serialize)]
struct WebGpuRuntimeDescription {
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

#[derive(Debug, Deserialize)]
struct WebGpuExecutionRequest {
    kind: String,
    #[serde(default)]
    entrypoint: String,
    #[serde(default)]
    workgroups: Option<[u32; 3]>,
    #[serde(default)]
    metadata: Value,
    #[serde(default = "default_output_words")]
    output_words: usize,
}

#[derive(Debug, Deserialize, Default)]
struct DispatchMetadata {
    #[serde(default)]
    shader_source: String,
    #[serde(default)]
    params_u32: Vec<u32>,
    #[serde(default)]
    result_encoding: ResultEncoding,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ResultEncoding {
    U32,
    F32,
}

impl Default for ResultEncoding {
    fn default() -> Self {
        Self::U32
    }
}

#[derive(Debug, Serialize)]
struct DispatchResponse {
    kind: &'static str,
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
    result_encoding: &'static str,
    output_words: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_f32: Option<Vec<f32>>,
    metadata: Value,
}

pub(crate) fn build_import(
    config: &WebGpuConfig,
    broker_addr: Option<&str>,
) -> Result<ImportObject<WebGpuHostState>> {
    let mut builder = ImportObjectBuilder::new(
        WEBGPU_IMPORT_MODULE,
        WebGpuHostState::new(config, broker_addr),
    )
    .context("creating WebGPU import module")?;

    builder
        .with_func::<(i32, i32), i32>(DESCRIBE_RUNTIME_FN, describe_runtime)
        .context("registering describe_runtime host function")?;
    builder
        .with_func::<(i32, i32, i32, i32, i32, i32, i32, i32), i32>(EXECUTE_FN, execute)
        .context("registering execute host function")?;

    Ok(builder.build())
}

fn describe_runtime(
    state: &mut WebGpuHostState,
    _instance: &mut Instance,
    frame: &mut CallingFrame,
    params: Vec<WasmValue>,
) -> Result<Vec<WasmValue>, CoreError> {
    let [out_ptr, out_cap] = params_as_i32::<2>(&params).map_err(host_error)?;
    let encoded = match state.broker_addr.as_deref() {
        Some(addr) => crate::broker::describe_runtime(addr),
        None => describe_runtime_payload(state),
    }
    .map_err(|err| host_error(err.to_string()))?;
    write_guest_bytes(frame, out_ptr, out_cap, &encoded)
        .map_err(|err| host_error(err.to_string()))?;

    let written = i32::try_from(encoded.len()).map_err(|_| host_error("response too large"))?;
    Ok(vec![WasmValue::from_i32(written)])
}

fn execute(
    state: &mut WebGpuHostState,
    _instance: &mut Instance,
    frame: &mut CallingFrame,
    params: Vec<WasmValue>,
) -> Result<Vec<WasmValue>, CoreError> {
    let [
        request_ptr,
        request_len,
        input_a_ptr,
        input_a_len,
        input_b_ptr,
        input_b_len,
        out_ptr,
        out_cap,
    ] = params_as_i32::<8>(&params).map_err(host_error)?;

    let request_bytes = read_guest_bytes(frame, request_ptr, request_len)
        .map_err(|err| host_error(err.to_string()))?;
    let input_a = read_guest_bytes(frame, input_a_ptr, input_a_len)
        .map_err(|err| host_error(err.to_string()))?;
    let input_b = read_guest_bytes(frame, input_b_ptr, input_b_len)
        .map_err(|err| host_error(err.to_string()))?;
    let encoded = match state.broker_addr.as_deref() {
        Some(addr) => crate::broker::execute(addr, &request_bytes, &input_a, &input_b),
        None => execute_payload(state, &request_bytes, &input_a, &input_b),
    }
    .map_err(|err| host_error(err.to_string()))?;

    write_guest_bytes(frame, out_ptr, out_cap, &encoded)
        .map_err(|err| host_error(err.to_string()))?;
    let written = i32::try_from(encoded.len()).map_err(|_| host_error("response too large"))?;
    Ok(vec![WasmValue::from_i32(written)])
}

fn runtime_description(state: &WebGpuHostState) -> WebGpuRuntimeDescription {
    if !state.config.enabled {
        return WebGpuRuntimeDescription {
            enabled: false,
            backend: state.config.backend.clone(),
            adapter_name: state.config.adapter_name.clone(),
            device_path: None,
            device_available: false,
            runtime_ready: false,
            runtime_error: None,
            max_buffer_size: state.config.max_buffer_size,
            max_bind_groups: state.config.max_bind_groups,
            force_fallback_adapter: state.config.force_fallback_adapter,
            required: state.config.required,
        };
    }

    // `describe_runtime` reports readiness details instead of failing the guest outright.
    match ensure_runtime(state) {
        Ok(runtime) => WebGpuRuntimeDescription {
            enabled: true,
            backend: runtime.backend.clone(),
            adapter_name: runtime.adapter_name.clone(),
            device_path: None,
            device_available: true,
            runtime_ready: true,
            runtime_error: None,
            max_buffer_size: effective_max_buffer_size(&state.config, runtime.as_ref()),
            max_bind_groups: effective_max_bind_groups(&state.config, runtime.as_ref()),
            force_fallback_adapter: state.config.force_fallback_adapter,
            required: state.config.required,
        },
        Err(err) => WebGpuRuntimeDescription {
            enabled: true,
            backend: state.config.backend.clone(),
            adapter_name: state.config.adapter_name.clone(),
            device_path: None,
            device_available: false,
            runtime_ready: false,
            runtime_error: Some(err.to_string()),
            max_buffer_size: state.config.max_buffer_size,
            max_bind_groups: state.config.max_bind_groups,
            force_fallback_adapter: state.config.force_fallback_adapter,
            required: state.config.required,
        },
    }
}

impl WebGpuHostState {
    fn new(config: &WebGpuConfig, broker_addr: Option<&str>) -> Self {
        Self {
            config: config.clone(),
            broker_addr: broker_addr
                .map(ToOwned::to_owned)
                .or_else(|| env::var(crate::broker::BROKER_ADDR_ENV).ok())
                .filter(|value| !value.is_empty()),
            runtime: Arc::new(OnceLock::new()),
        }
    }

    pub(crate) fn direct(config: &WebGpuConfig) -> Self {
        Self {
            config: config.clone(),
            broker_addr: None,
            runtime: Arc::new(OnceLock::new()),
        }
    }
}

pub(crate) fn describe_runtime_payload(state: &WebGpuHostState) -> Result<Vec<u8>> {
    serde_json::to_vec(&runtime_description(state)).context("serializing runtime description")
}

pub(crate) fn execute_payload(
    state: &WebGpuHostState,
    request_bytes: &[u8],
    input_a: &[u8],
    input_b: &[u8],
) -> Result<Vec<u8>> {
    validate_runtime_access(&state.config)
        .map_err(|err| anyhow!("runtime access denied: {err:?}"))?;
    let request: WebGpuExecutionRequest =
        serde_json::from_slice(request_bytes).context("parsing execution request")?;
    let runtime = ensure_runtime(state)?;

    match request.kind.as_str() {
        "compute.dispatch" => {
            execute_dispatch(&state.config, runtime.as_ref(), &request, input_a, input_b)
        }
        other => Err(anyhow!("unsupported webgpu execution kind: {other}")),
    }
}

fn execute_dispatch(
    config: &WebGpuConfig,
    runtime: &WebGpuRuntime,
    request: &WebGpuExecutionRequest,
    input_a: &[u8],
    input_b: &[u8],
) -> Result<Vec<u8>> {
    let metadata = parse_dispatch_metadata(&request.metadata)?;
    if metadata.shader_source.trim().is_empty() {
        bail!("metadata.shader_source must contain WGSL source for compute.dispatch");
    }

    let workgroups = request.workgroups.unwrap_or([1, 1, 1]);
    let invocations = workgroups
        .into_iter()
        .fold(1u32, |acc, value| acc.saturating_mul(value.max(1)));
    let output_size = request
        .output_words
        .checked_mul(4)
        .context("output_words overflow")?;
    if output_size == 0 {
        bail!("output_words must be greater than zero");
    }
    let output_size = u64::try_from(output_size).context("output size is too large")?;
    let effective_max_buffer_size = effective_max_buffer_size(config, runtime);
    if output_size > effective_max_buffer_size {
        bail!(
            "requested output buffer ({output_size} bytes) exceeds the effective max buffer size ({effective_max_buffer_size} bytes)"
        );
    }

    let output_bytes = runtime.run_compute(
        &metadata.shader_source,
        dispatch_entrypoint(request),
        workgroups,
        input_a,
        input_b,
        &uniform_bytes(&metadata.params_u32),
        output_size,
    )?;

    let output_words = bytes_to_u32_words(&output_bytes)?;
    let output_f32 = match metadata.result_encoding {
        ResultEncoding::U32 => None,
        ResultEncoding::F32 => Some(bytes_to_f32(&output_bytes)?),
    };
    let checksum = checksum(&output_bytes);

    let response = DispatchResponse {
        kind: "compute.dispatch",
        entrypoint: dispatch_entrypoint(request).to_string(),
        backend: runtime.backend.clone(),
        adapter_name: runtime.adapter_name.clone(),
        device_path: None,
        workgroups,
        invocations,
        checksum,
        input_a_len: input_a.len(),
        input_b_len: input_b.len(),
        output_bytes: output_bytes.len(),
        result_encoding: metadata.result_encoding.as_str(),
        output_words,
        output_f32,
        metadata: request.metadata.clone(),
    };

    serde_json::to_vec(&response).context("serializing dispatch response")
}

impl WebGpuRuntime {
    fn new(config: &WebGpuConfig) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: requested_backends(&config.backend)?,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: config.force_fallback_adapter,
        }))
        .context("requesting a GPU adapter via wgpu")?;
        let adapter_info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("runwasi-webgpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .context("requesting a GPU device via wgpu")?;

        Ok(Self {
            device,
            queue,
            backend: backend_name(adapter_info.backend).to_string(),
            adapter_name: adapter_info.name,
            max_buffer_size: limits.max_buffer_size,
            max_bind_groups: limits.max_bind_groups,
        })
    }

    fn run_compute(
        &self,
        shader_source: &str,
        entrypoint: &str,
        workgroups: [u32; 3],
        input_a: &[u8],
        input_b: &[u8],
        params: &[u8],
        output_size: u64,
    ) -> Result<Vec<u8>> {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("runwasi-webgpu-shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.to_string().into()),
            });

        // The current generic ABI intentionally keeps one fixed bind-group layout so guest
        // modules can submit raw tensors without negotiating resource handles first.
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("runwasi-webgpu-bind-group-layout"),
                    entries: &[
                        bind_group_layout_entry(
                            0,
                            wgpu::BufferBindingType::Storage { read_only: true },
                        ),
                        bind_group_layout_entry(
                            1,
                            wgpu::BufferBindingType::Storage { read_only: true },
                        ),
                        bind_group_layout_entry(
                            2,
                            wgpu::BufferBindingType::Storage { read_only: false },
                        ),
                        bind_group_layout_entry(3, wgpu::BufferBindingType::Uniform),
                    ],
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("runwasi-webgpu-pipeline-layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("runwasi-webgpu-compute-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entrypoint),
                cache: None,
                compilation_options: Default::default(),
            });

        let input_a_buffer = buffer_with_contents(
            &self.device,
            "runwasi-webgpu-input-a",
            &storage_bytes(input_a),
            wgpu::BufferUsages::STORAGE,
        );
        let input_b_buffer = buffer_with_contents(
            &self.device,
            "runwasi-webgpu-input-b",
            &storage_bytes(input_b),
            wgpu::BufferUsages::STORAGE,
        );
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runwasi-webgpu-output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buffer = buffer_with_contents(
            &self.device,
            "runwasi-webgpu-params",
            params,
            wgpu::BufferUsages::UNIFORM,
        );
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("runwasi-webgpu-readback"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("runwasi-webgpu-bind-group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("runwasi-webgpu-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("runwasi-webgpu-compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = self.device.poll(wgpu::Maintain::wait());
        receiver
            .recv()
            .context("waiting for GPU readback")?
            .context("mapping GPU readback buffer")?;

        let data = slice.get_mapped_range().to_vec();
        let _ = slice;
        readback_buffer.unmap();
        Ok(data)
    }
}

fn ensure_runtime(state: &WebGpuHostState) -> Result<Arc<WebGpuRuntime>> {
    if let Some(runtime) = state.runtime.get() {
        return Ok(runtime.clone());
    }

    let runtime = Arc::new(WebGpuRuntime::new(&state.config)?);
    match state.runtime.set(runtime.clone()) {
        Ok(()) => Ok(runtime),
        Err(_) => state
            .runtime
            .get()
            .cloned()
            .ok_or_else(|| anyhow!("webgpu runtime initialization raced and left no runtime")),
    }
}

fn validate_runtime_access(config: &WebGpuConfig) -> Result<(), CoreError> {
    if !config.enabled {
        return Err(host_error("WEBGPU_ENABLED=0 blocks webgpu.execute"));
    }

    Ok(())
}

fn parse_dispatch_metadata(metadata: &Value) -> Result<DispatchMetadata> {
    if metadata.is_null() {
        return Ok(DispatchMetadata::default());
    }

    serde_json::from_value(metadata.clone()).context("parsing dispatch metadata")
}

fn dispatch_entrypoint(request: &WebGpuExecutionRequest) -> &str {
    if request.entrypoint.is_empty() {
        "main"
    } else {
        request.entrypoint.as_str()
    }
}

fn effective_max_buffer_size(config: &WebGpuConfig, runtime: &WebGpuRuntime) -> u64 {
    runtime.max_buffer_size.min(config.max_buffer_size)
}

fn effective_max_bind_groups(config: &WebGpuConfig, runtime: &WebGpuRuntime) -> u32 {
    runtime.max_bind_groups.min(config.max_bind_groups)
}

fn bind_group_layout_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_with_contents(
    device: &wgpu::Device,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let size = u64::try_from(contents.len()).unwrap_or(4).max(4);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    buffer.slice(..).get_mapped_range_mut()[..contents.len()].copy_from_slice(contents);
    buffer.unmap();
    buffer
}

fn storage_bytes(bytes: &[u8]) -> Vec<u8> {
    // Storage bindings cannot be zero-sized, so empty guest buffers are padded to one word.
    if bytes.is_empty() {
        vec![0; 4]
    } else {
        bytes.to_vec()
    }
}

fn uniform_bytes(values: &[u32]) -> Vec<u8> {
    // Keep the uniform payload aligned for the simple `vec4<u32>`-style parameter block.
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

fn bytes_to_u32_words(bytes: &[u8]) -> Result<Vec<u32>> {
    if bytes.len() % 4 != 0 {
        bail!("GPU output buffer size is not divisible by 4");
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap_or([0; 4])))
        .collect())
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

fn requested_backends(value: &str) -> Result<wgpu::Backends> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(wgpu::Backends::all()),
        "vulkan" => Ok(wgpu::Backends::VULKAN),
        "metal" => Ok(wgpu::Backends::METAL),
        "dx12" => Ok(wgpu::Backends::DX12),
        "gl" => Ok(wgpu::Backends::GL),
        "browser_webgpu" => Ok(wgpu::Backends::BROWSER_WEBGPU),
        other => bail!("unsupported WEBGPU_BACKEND value: {other}"),
    }
}

fn backend_name(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Empty => "empty",
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Gl => "gl",
        wgpu::Backend::BrowserWebGpu => "browser_webgpu",
    }
}

fn checksum(bytes: &[u8]) -> u64 {
    mix_bytes(0xcbf2_9ce4_8422_2325u64, bytes)
}

fn params_as_i32<const N: usize>(params: &[WasmValue]) -> Result<[i32; N], &'static str> {
    if params.len() != N {
        return Err("unexpected host function argument count");
    }

    let mut values = [0i32; N];
    for (slot, value) in values.iter_mut().zip(params.iter()) {
        *slot = value.to_i32();
    }

    Ok(values)
}

fn read_guest_bytes(frame: &mut CallingFrame, offset: i32, len: i32) -> Result<Vec<u8>> {
    let offset = u32::try_from(offset).context("negative guest offset")?;
    let len = u32::try_from(len).context("negative guest length")?;
    let memory = frame
        .memory_mut(0)
        .ok_or_else(|| anyhow!("guest memory is not available"))?;

    memory
        .get_data(offset, len)
        .context("reading guest memory for host function input")
}

fn write_guest_bytes(frame: &mut CallingFrame, offset: i32, cap: i32, data: &[u8]) -> Result<()> {
    let offset = u32::try_from(offset).context("negative guest output offset")?;
    let cap = usize::try_from(cap).context("negative guest output capacity")?;
    if data.len() > cap {
        return Err(anyhow!("guest output buffer is too small"));
    }

    let mut memory = frame
        .memory_mut(0)
        .ok_or_else(|| anyhow!("guest memory is not available"))?;
    memory
        .set_data(data, offset)
        .context("writing host function output into guest memory")
}

fn mix_bytes(mut state: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(1_099_511_628_211);
    }
    state
}

fn host_error(message: impl Into<String>) -> CoreError {
    log::error!("webgpu host function error: {}", message.into());
    CoreError::Execution(CoreExecutionError::HostFuncFailed)
}

fn default_output_words() -> usize {
    8
}

impl ResultEncoding {
    fn as_str(self) -> &'static str {
        match self {
            Self::U32 => "u32",
            Self::F32 => "f32",
        }
    }
}
