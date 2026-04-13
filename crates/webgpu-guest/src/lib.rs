use std::error::Error as StdError;
use std::fmt;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const DEFAULT_OUTPUT_CAPACITY: usize = 16 * 1024;

#[derive(Clone, Debug, Deserialize)]
pub struct RuntimeDescription {
    pub enabled: bool,
    pub backend: String,
    pub adapter_name: String,
    pub device_available: bool,
    pub runtime_ready: bool,
    pub runtime_error: Option<String>,
    pub max_buffer_size: u64,
    pub max_bind_groups: u32,
    pub force_fallback_adapter: bool,
    pub required: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DispatchResponse {
    pub kind: String,
    pub entrypoint: String,
    pub backend: String,
    pub adapter_name: String,
    pub workgroups: [u32; 3],
    pub invocations: u32,
    pub checksum: u64,
    pub input_a_len: usize,
    pub input_b_len: usize,
    pub output_bytes: usize,
    pub result_encoding: String,
    pub output_words: Vec<u32>,
    pub output_f32: Option<Vec<f32>>,
    pub metadata: Value,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultEncoding {
    U32,
    F32,
}

impl Default for ResultEncoding {
    fn default() -> Self {
        Self::U32
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ComputeDispatch {
    kind: &'static str,
    entrypoint: String,
    workgroups: Option<[u32; 3]>,
    output_words: usize,
    metadata: Value,
}

impl ComputeDispatch {
    pub fn new(
        entrypoint: impl Into<String>,
        shader_source: impl Into<String>,
        output_words: usize,
    ) -> Self {
        let mut metadata = Map::new();
        metadata.insert(
            "shader_source".to_string(),
            Value::String(shader_source.into()),
        );
        metadata.insert("params_u32".to_string(), Value::Array(Vec::new()));
        metadata.insert(
            "result_encoding".to_string(),
            Value::String("u32".to_string()),
        );

        Self {
            kind: "compute.dispatch",
            entrypoint: entrypoint.into(),
            workgroups: None,
            output_words,
            metadata: Value::Object(metadata),
        }
    }

    pub fn workgroups(mut self, workgroups: [u32; 3]) -> Self {
        self.workgroups = Some(workgroups);
        self
    }

    pub fn params_u32<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = u32>,
    {
        if let Some(metadata) = self.metadata.as_object_mut() {
            metadata.insert(
                "params_u32".to_string(),
                Value::Array(values.into_iter().map(Value::from).collect()),
            );
        }
        self
    }

    pub fn result_encoding(mut self, encoding: ResultEncoding) -> Self {
        if let Some(metadata) = self.metadata.as_object_mut() {
            metadata.insert(
                "result_encoding".to_string(),
                Value::String(match encoding {
                    ResultEncoding::U32 => "u32".to_string(),
                    ResultEncoding::F32 => "f32".to_string(),
                }),
            );
        }
        self
    }

    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        if let Some(metadata) = self.metadata.as_object_mut() {
            metadata.insert(key.into(), value.into());
        }
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Client {
    output_capacity: usize,
}

impl Client {
    pub const fn new(output_capacity: usize) -> Self {
        Self { output_capacity }
    }

    pub fn describe_runtime(self) -> Result<RuntimeDescription, Error> {
        let mut output = vec![0u8; self.output_capacity];
        let written = unsafe {
            webgpu_describe_runtime(
                output.as_mut_ptr(),
                i32::try_from(output.len()).map_err(|_| Error::SizeOverflow("output buffer"))?,
            )
        };
        let written = check_host_result(written, "webgpu.describe_runtime")?;
        parse_json("runtime description", &output[..written])
    }

    pub fn execute_raw<TReq: Serialize>(
        self,
        request: &TReq,
        input_a: &[u8],
        input_b: &[u8],
    ) -> Result<Vec<u8>, Error> {
        let request_bytes =
            serde_json::to_vec(request).map_err(|err| Error::Json("request json", err))?;
        let mut output = vec![0u8; self.output_capacity];

        let written = unsafe {
            webgpu_execute(
                request_bytes.as_ptr(),
                i32::try_from(request_bytes.len()).map_err(|_| Error::SizeOverflow("request"))?,
                input_a.as_ptr(),
                i32::try_from(input_a.len()).map_err(|_| Error::SizeOverflow("input_a"))?,
                input_b.as_ptr(),
                i32::try_from(input_b.len()).map_err(|_| Error::SizeOverflow("input_b"))?,
                output.as_mut_ptr(),
                i32::try_from(output.len()).map_err(|_| Error::SizeOverflow("output buffer"))?,
            )
        };

        let written = check_host_result(written, "webgpu.execute")?;
        output.truncate(written);
        Ok(output)
    }

    pub fn execute<TReq: Serialize, TRes: DeserializeOwned>(
        self,
        request: &TReq,
        input_a: &[u8],
        input_b: &[u8],
    ) -> Result<TRes, Error> {
        let bytes = self.execute_raw(request, input_a, input_b)?;
        parse_json("execute response", &bytes)
    }
}

impl Default for Client {
    fn default() -> Self {
        Self::new(DEFAULT_OUTPUT_CAPACITY)
    }
}

#[derive(Debug)]
pub enum Error {
    HostFunctionFailed(&'static str),
    InvalidLength(&'static str),
    Json(&'static str, serde_json::Error),
    SizeOverflow(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostFunctionFailed(name) => write!(f, "{name} returned an error"),
            Self::InvalidLength(name) => write!(f, "{name} returned an invalid response length"),
            Self::Json(name, err) => write!(f, "invalid {name}: {err}"),
            Self::SizeOverflow(name) => write!(f, "{name} is too large for the host ABI"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Json(_, err) => Some(err),
            _ => None,
        }
    }
}

pub fn describe_runtime() -> Result<RuntimeDescription, Error> {
    Client::default().describe_runtime()
}

pub fn execute_raw<TReq: Serialize>(
    request: &TReq,
    input_a: &[u8],
    input_b: &[u8],
) -> Result<Vec<u8>, Error> {
    Client::default().execute_raw(request, input_a, input_b)
}

pub fn execute<TReq: Serialize, TRes: DeserializeOwned>(
    request: &TReq,
    input_a: &[u8],
    input_b: &[u8],
) -> Result<TRes, Error> {
    Client::default().execute(request, input_a, input_b)
}

pub fn bytes_from_f32_slice(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn parse_json<T: DeserializeOwned>(name: &'static str, bytes: &[u8]) -> Result<T, Error> {
    serde_json::from_slice(bytes).map_err(|err| Error::Json(name, err))
}

fn check_host_result(written: i32, name: &'static str) -> Result<usize, Error> {
    if written < 0 {
        return Err(Error::HostFunctionFailed(name));
    }
    usize::try_from(written).map_err(|_| Error::InvalidLength(name))
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
