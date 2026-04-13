use std::collections::HashMap;
use std::env;
#[cfg(all(feature = "plugin", not(target_env = "musl")))]
use std::str::FromStr;

use anyhow::{Context, Result};
use containerd_shim_wasm::sandbox::Sandbox;
use containerd_shim_wasm::sandbox::context::{Entrypoint, RuntimeContext};
use containerd_shim_wasm::shim::{InstanceGuard, Shim, Version, version};
use containerd_shimkit::sandbox::InstanceConfig;
use oci_spec::runtime::Spec;
use wasmedge_sdk::AsInstance;
use wasmedge_sdk::config::{CommonConfigOptions, Config, ConfigBuilder};
#[cfg(all(feature = "plugin", not(target_env = "musl")))]
use wasmedge_sdk::plugin::NNPreload;
#[cfg(all(feature = "plugin", not(target_env = "musl")))]
use wasmedge_sdk::plugin::PluginManager;
use wasmedge_sdk::vm::SyncInst;
use wasmedge_sdk::wasi::WasiModule;
use wasmedge_sdk::{Module, Store, Vm};

use crate::host;
use crate::middleware::WebGpuMiddleware;

pub struct WasmEdgeWebGpuShim;

pub struct WasmEdgeWebGpuSandbox {
    config: Config,
}

impl Default for WasmEdgeWebGpuSandbox {
    fn default() -> Self {
        let config = ConfigBuilder::new(CommonConfigOptions::default())
            .build()
            .expect("failed to create config");
        Self { config }
    }
}

impl Shim for WasmEdgeWebGpuShim {
    fn name() -> &'static str {
        "webgpu"
    }

    fn version() -> Version {
        version!()
    }

    async fn prepare_instance(
        id: &str,
        cfg: &InstanceConfig,
        spec: &mut Spec,
    ) -> Result<Option<Box<dyn InstanceGuard>>> {
        crate::broker::prepare_instance(id, cfg, spec).await
    }

    type Sandbox = WasmEdgeWebGpuSandbox;
}

impl Sandbox for WasmEdgeWebGpuSandbox {
    async fn run_wasi(&self, ctx: &impl RuntimeContext) -> Result<i32> {
        let args = ctx.args();
        let broker_addr = env_value(ctx.envs(), crate::broker::BROKER_ADDR_ENV);
        let middleware =
            WebGpuMiddleware::new(ctx.envs()).context("configuring WebGPU middleware")?;
        let envs = middleware.guest_envs(ctx.envs());
        let Entrypoint {
            source,
            func,
            arg0: _,
            name,
        } = ctx.entrypoint();

        log::debug!("initializing WasmEdge WebGPU runtime");
        log::info!(
            "webgpu middleware enabled={}, backend={}, adapter={}, device_path={:?}",
            middleware.config().enabled,
            middleware.config().backend,
            middleware.config().adapter_name,
            middleware.config().device_path
        );

        let prefix = "WASMEDGE_";
        // Mirror explicit WasmEdge host configuration into the process environment before
        // instantiating the VM so native WasmEdge options still behave as expected.
        for env in envs.iter() {
            if let Some((key, value)) = env.split_once('=') {
                if key.starts_with(prefix) {
                    unsafe {
                        env::set_var(key, value);
                    }
                }
            }
        }

        #[cfg(all(feature = "plugin", not(target_env = "musl")))]
        let mut wasi_nn = {
            PluginManager::load(None)?;
            match env::var("WASMEDGE_WASINN_PRELOAD") {
                Ok(value) => PluginManager::nn_preload(vec![NNPreload::from_str(value.as_str())?]),
                Err(_) => log::debug!("No specific nn_preload parameter for wasi_nn plugin"),
            }

            // Load the wasi_nn plugin manually as a workaround.
            // It should call auto_detect_plugins after the issue is fixed.
            PluginManager::names()
                .contains(&"wasi_nn".to_string())
                .then(PluginManager::load_plugin_wasi_nn)
                .transpose()?
        };

        let mut instances: HashMap<String, &mut dyn SyncInst> = HashMap::new();
        #[cfg(all(feature = "plugin", not(target_env = "musl")))]
        if let Some(ref mut nn) = wasi_nn {
            let nn_name = nn.name().unwrap_or_else(|| "wasi_nn".to_string());
            instances.insert(nn_name, nn);
        }

        let mut webgpu_host = host::build_import(middleware.config(), broker_addr)
            .context("creating WebGPU host import module")?;
        let webgpu_name = webgpu_host.name().unwrap_or_else(|| "webgpu".to_string());
        instances.insert(webgpu_name, &mut webgpu_host);

        // Keep filesystem exposure intentionally simple for now. Tight isolation can be layered
        // back in once the real GPU path and ABI are stable.
        let mut wasi_module = WasiModule::create(
            Some(args.iter().map(String::as_str).collect()),
            Some(envs.iter().map(String::as_str).collect()),
            Some(vec!["/:/"]),
        )?;
        instances.insert(wasi_module.name().to_string(), wasi_module.as_mut());

        let wasm_bytes = source.as_bytes()?;
        let module = Module::from_bytes(Some(&self.config), &wasm_bytes)?;
        let mod_name = name.unwrap_or_else(|| "main".to_string());
        {
            let store =
                Store::new(Some(&self.config), instances).context("creating WasmEdge store")?;
            let mut vm = Vm::new(store);
            vm.register_module(Some(&mod_name), module)
                .context("registering module")?;

            log::debug!("running with method {func:?}");
            vm.run_func(Some(&mod_name), func, vec![])?;
        }

        Ok(wasi_module.exit_code() as i32)
    }
}

fn env_value<'a>(envs: &'a [String], key: &str) -> Option<&'a str> {
    envs.iter()
        .find_map(|env| env.strip_prefix(&format!("{key}=")))
}
