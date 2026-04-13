pub(crate) const BROKER_ADDR_ENV: &str = "WEBGPU_BROKER_ADDR";

#[cfg(target_os = "linux")]
mod imp {
    use std::io::{Read, Write};
    use std::os::linux::net::SocketAddrExt;
    use std::os::unix::net::{SocketAddr, UnixListener, UnixStream};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread::{self, JoinHandle};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use anyhow::{Context, Result, anyhow, bail};
    use containerd_shim_wasm::shim::InstanceGuard;
    use containerd_shimkit::sandbox::InstanceConfig;
    use oci_spec::runtime::{Process, Spec};

    use crate::broker::BROKER_ADDR_ENV;
    use crate::host::{self, WebGpuHostState};
    use crate::middleware::WebGpuMiddleware;

    const OP_DESCRIBE_RUNTIME: u8 = 1;
    const OP_EXECUTE: u8 = 2;
    const STATUS_OK: u8 = 0;
    const STATUS_ERR: u8 = 1;

    static BROKER_COUNTER: AtomicU64 = AtomicU64::new(0);

    pub async fn prepare_instance(
        id: &str,
        _cfg: &InstanceConfig,
        spec: &mut Spec,
    ) -> Result<Option<Box<dyn InstanceGuard>>> {
        let envs = spec
            .process()
            .as_ref()
            .and_then(|process| process.env().as_ref())
            .cloned()
            .unwrap_or_default();
        let middleware = WebGpuMiddleware::new(&envs)?;
        let state = WebGpuHostState::direct(middleware.config());
        let broker = WebGpuBroker::start(id, state)?;

        let process = spec.process_mut().get_or_insert_with(Process::default);
        let process_envs = process.env_mut().get_or_insert_with(Vec::new);
        ensure_env(process_envs, BROKER_ADDR_ENV, broker.name().to_string());

        Ok(Some(Box::new(broker)))
    }

    pub fn describe_runtime(name: &str) -> Result<Vec<u8>> {
        let mut stream = connect(name)?;
        write_u8(&mut stream, OP_DESCRIBE_RUNTIME)?;
        read_response(&mut stream)
    }

    pub fn execute(name: &str, request: &[u8], input_a: &[u8], input_b: &[u8]) -> Result<Vec<u8>> {
        let mut stream = connect(name)?;
        write_u8(&mut stream, OP_EXECUTE)?;
        write_blob(&mut stream, request)?;
        write_blob(&mut stream, input_a)?;
        write_blob(&mut stream, input_b)?;
        read_response(&mut stream)
    }

    struct WebGpuBroker {
        name: String,
        shutdown: Arc<AtomicBool>,
        thread: Mutex<Option<JoinHandle<()>>>,
    }

    impl WebGpuBroker {
        fn start(id: &str, state: WebGpuHostState) -> Result<Self> {
            let name = broker_name(id);
            let addr = SocketAddr::from_abstract_name(name.as_bytes())
                .context("building abstract Unix socket address for webgpu broker")?;
            let listener =
                UnixListener::bind_addr(&addr).context("binding webgpu broker socket")?;
            listener
                .set_nonblocking(true)
                .context("marking webgpu broker socket non-blocking")?;

            let shutdown = Arc::new(AtomicBool::new(false));
            let shutdown_thread = shutdown.clone();
            let thread = thread::Builder::new()
                .name(format!("webgpu-broker-{id}"))
                .spawn(move || serve(listener, shutdown_thread, state))
                .context("spawning webgpu broker thread")?;

            Ok(Self {
                name,
                shutdown,
                thread: Mutex::new(Some(thread)),
            })
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    impl Drop for WebGpuBroker {
        fn drop(&mut self) {
            self.shutdown.store(true, Ordering::Relaxed);
            if let Ok(thread) = self.thread.get_mut() {
                if let Some(thread) = thread.take() {
                    let _ = thread.join();
                }
            }
        }
    }

    fn serve(listener: UnixListener, shutdown: Arc<AtomicBool>, state: WebGpuHostState) {
        while !shutdown.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    if let Err(err) = handle_client(&mut stream, &state) {
                        log::error!("webgpu broker client request failed: {err}");
                    }
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(25));
                }
                Err(err) => {
                    log::error!("webgpu broker accept failed: {err}");
                    thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }

    fn handle_client(stream: &mut UnixStream, state: &WebGpuHostState) -> Result<()> {
        let op = read_u8(stream)?;
        let result = match op {
            OP_DESCRIBE_RUNTIME => host::describe_runtime_payload(state),
            OP_EXECUTE => {
                let request = read_blob(stream)?;
                let input_a = read_blob(stream)?;
                let input_b = read_blob(stream)?;
                host::execute_payload(state, &request, &input_a, &input_b)
            }
            other => Err(anyhow!("unsupported webgpu broker opcode: {other}")),
        };

        match result {
            Ok(payload) => write_response(stream, STATUS_OK, &payload),
            Err(err) => write_response(stream, STATUS_ERR, err.to_string().as_bytes()),
        }
    }

    fn connect(name: &str) -> Result<UnixStream> {
        let addr = SocketAddr::from_abstract_name(name.as_bytes())
            .context("building broker client socket address")?;
        UnixStream::connect_addr(&addr).context("connecting to webgpu broker")
    }

    fn write_response(stream: &mut UnixStream, status: u8, payload: &[u8]) -> Result<()> {
        write_u8(stream, status)?;
        write_blob(stream, payload)?;
        Ok(())
    }

    fn read_response(stream: &mut UnixStream) -> Result<Vec<u8>> {
        let status = read_u8(stream)?;
        let payload = read_blob(stream)?;
        match status {
            STATUS_OK => Ok(payload),
            STATUS_ERR => bail!("{}", String::from_utf8_lossy(&payload)),
            other => bail!("unexpected webgpu broker status: {other}"),
        }
    }

    fn write_u8(stream: &mut UnixStream, value: u8) -> Result<()> {
        stream.write_all(&[value]).context("writing broker byte")
    }

    fn read_u8(stream: &mut UnixStream) -> Result<u8> {
        let mut buf = [0u8; 1];
        stream.read_exact(&mut buf).context("reading broker byte")?;
        Ok(buf[0])
    }

    fn write_blob(stream: &mut UnixStream, payload: &[u8]) -> Result<()> {
        let len = u32::try_from(payload.len()).context("broker payload too large")?;
        stream
            .write_all(&len.to_le_bytes())
            .context("writing broker payload length")?;
        stream
            .write_all(payload)
            .context("writing broker payload bytes")?;
        Ok(())
    }

    fn read_blob(stream: &mut UnixStream) -> Result<Vec<u8>> {
        let mut len_buf = [0u8; 4];
        stream
            .read_exact(&mut len_buf)
            .context("reading broker payload length")?;
        let len = usize::try_from(u32::from_le_bytes(len_buf))
            .context("invalid broker payload length")?;
        let mut payload = vec![0u8; len];
        stream
            .read_exact(&mut payload)
            .context("reading broker payload bytes")?;
        Ok(payload)
    }

    fn ensure_env(envs: &mut Vec<String>, key: &str, value: String) {
        if envs.iter().all(|env| !env.starts_with(&format!("{key}="))) {
            envs.push(format!("{key}={value}"));
        }
    }

    fn broker_name(id: &str) -> String {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let counter = BROKER_COUNTER.fetch_add(1, Ordering::Relaxed);
        let id = sanitize(id).chars().take(16).collect::<String>();
        format!("runwasi-webgpu-{id}-{ts:x}-{counter:x}")
    }

    fn sanitize(value: &str) -> String {
        value
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                    ch
                } else {
                    '-'
                }
            })
            .collect()
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use anyhow::{Result, bail};
    use containerd_shim_wasm::shim::InstanceGuard;
    use containerd_shimkit::sandbox::InstanceConfig;
    use oci_spec::runtime::Spec;

    pub async fn prepare_instance(
        _id: &str,
        _cfg: &InstanceConfig,
        _spec: &mut Spec,
    ) -> Result<Option<Box<dyn InstanceGuard>>> {
        Ok(None)
    }

    pub fn describe_runtime(_name: &str) -> Result<Vec<u8>> {
        bail!("webgpu broker is only supported on Linux")
    }

    pub fn execute(
        _name: &str,
        _request: &[u8],
        _input_a: &[u8],
        _input_b: &[u8],
    ) -> Result<Vec<u8>> {
        bail!("webgpu broker is only supported on Linux")
    }
}

pub use imp::{describe_runtime, execute, prepare_instance};
