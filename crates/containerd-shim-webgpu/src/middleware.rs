use std::path::Path;

use anyhow::{Result, bail};

const HOST_ONLY_GUEST_ENV_KEYS: &[&str] = &[
    "WEBGPU_DEVICE_PATH",
    "WGPU_DEVICE_PATH",
    crate::broker::BROKER_ADDR_ENV,
];
const DEFAULT_WEBGPU_ADAPTER_NAME: &str = "default";
const DEFAULT_WEBGPU_BACKEND: &str = "auto";
const DEFAULT_MAX_BUFFER_SIZE: u64 = 128 * 1024 * 1024;
const DEFAULT_MAX_BIND_GROUPS: u32 = 4;
const KNOWN_GPU_DEVICE_PATHS: &[&str] = &[
    "/dev/dri/renderD128",
    "/dev/dri/card0",
    "/dev/nvidia0",
    "/dev/kfd",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebGpuConfig {
    pub enabled: bool,
    pub backend: String,
    pub adapter_name: String,
    pub device_path: Option<String>,
    pub max_buffer_size: u64,
    pub max_bind_groups: u32,
    pub force_fallback_adapter: bool,
    pub required: bool,
}

impl WebGpuConfig {
    fn from_envs(envs: &[String]) -> Self {
        // A Linux device path still helps operationally, but native `wgpu` can resolve a real
        // adapter even when the backend is selected purely by platform, such as Metal on macOS.
        let device_path = env_value(envs, "WEBGPU_DEVICE_PATH")
            .map(ToOwned::to_owned)
            .filter(|value| !value.is_empty())
            .or_else(detect_device_path);

        let backend = env_value(envs, "WEBGPU_BACKEND")
            .or_else(|| env_value(envs, "WGPU_BACKEND"))
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| detect_backend(device_path.as_deref()).to_string());

        let adapter_name = env_value(envs, "WEBGPU_ADAPTER_NAME")
            .or_else(|| env_value(envs, "WGPU_ADAPTER_NAME"))
            .unwrap_or(DEFAULT_WEBGPU_ADAPTER_NAME)
            .to_string();

        Self {
            enabled: env_bool(envs, "WEBGPU_ENABLED").unwrap_or(true),
            backend,
            adapter_name,
            device_path,
            max_buffer_size: env_u64(envs, "WEBGPU_MAX_BUFFER_SIZE")
                .unwrap_or(DEFAULT_MAX_BUFFER_SIZE),
            max_bind_groups: env_u32(envs, "WEBGPU_MAX_BIND_GROUPS")
                .unwrap_or(DEFAULT_MAX_BIND_GROUPS),
            force_fallback_adapter: env_bool(envs, "WEBGPU_FORCE_FALLBACK_ADAPTER")
                .unwrap_or(false),
            required: env_bool(envs, "WEBGPU_REQUIRED").unwrap_or(false),
        }
    }

    fn validate(&self) -> Result<()> {
        if self.required && !self.enabled {
            bail!("WEBGPU_REQUIRED=1 cannot be combined with WEBGPU_ENABLED=0");
        }

        Ok(())
    }
}

pub struct WebGpuMiddleware {
    config: WebGpuConfig,
}

impl WebGpuMiddleware {
    pub fn new(envs: &[String]) -> Result<Self> {
        let config = WebGpuConfig::from_envs(envs);
        config.validate()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &WebGpuConfig {
        &self.config
    }

    pub fn guest_envs(&self, envs: &[String]) -> Vec<String> {
        let mut merged = envs
            .iter()
            .filter(|env| {
                HOST_ONLY_GUEST_ENV_KEYS
                    .iter()
                    .all(|key| !env.starts_with(&format!("{key}=")))
            })
            .cloned()
            .collect::<Vec<_>>();

        // Mirror the normalized host configuration back into the guest so a wasm workload can
        // inspect the effective backend choice without re-implementing host-side detection.
        ensure_env(&mut merged, "WEBGPU_ENABLED", bool_env(self.config.enabled));
        ensure_env(&mut merged, "WEBGPU_BACKEND", self.config.backend.clone());
        ensure_env(
            &mut merged,
            "WEBGPU_ADAPTER_NAME",
            self.config.adapter_name.clone(),
        );
        ensure_env(
            &mut merged,
            "WEBGPU_DEVICE_AVAILABLE",
            bool_env(self.config.enabled),
        );
        ensure_env(
            &mut merged,
            "WEBGPU_MAX_BUFFER_SIZE",
            self.config.max_buffer_size.to_string(),
        );
        ensure_env(
            &mut merged,
            "WEBGPU_MAX_BIND_GROUPS",
            self.config.max_bind_groups.to_string(),
        );
        ensure_env(
            &mut merged,
            "WEBGPU_FORCE_FALLBACK_ADAPTER",
            bool_env(self.config.force_fallback_adapter),
        );
        ensure_env(
            &mut merged,
            "WEBGPU_REQUIRED",
            bool_env(self.config.required),
        );
        ensure_env(&mut merged, "WGPU_BACKEND", self.config.backend.clone());
        ensure_env(
            &mut merged,
            "WGPU_ADAPTER_NAME",
            self.config.adapter_name.clone(),
        );

        merged
    }
}

fn ensure_env(envs: &mut Vec<String>, key: &str, value: String) {
    if env_value(envs, key).is_none() {
        envs.push(format!("{key}={value}"));
    }
}

fn env_value<'a>(envs: &'a [String], key: &str) -> Option<&'a str> {
    envs.iter()
        .find_map(|env| env.strip_prefix(&format!("{key}=")))
}

fn env_bool(envs: &[String], key: &str) -> Option<bool> {
    env_value(envs, key).and_then(parse_bool)
}

fn env_u32(envs: &[String], key: &str) -> Option<u32> {
    env_value(envs, key).and_then(|value| value.parse().ok())
}

fn env_u64(envs: &[String], key: &str) -> Option<u64> {
    env_value(envs, key).and_then(|value| value.parse().ok())
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn detect_device_path() -> Option<String> {
    KNOWN_GPU_DEVICE_PATHS
        .iter()
        .find(|path| Path::new(path).exists())
        .map(|path| (*path).to_string())
}

fn detect_backend(device_path: Option<&str>) -> &'static str {
    match device_path {
        Some(path) if path.starts_with("/dev/dri") => "vulkan",
        Some(path) if path.starts_with("/dev/nvidia") => "vulkan",
        Some(path) if path.starts_with("/dev/kfd") => "vulkan",
        Some(_) => DEFAULT_WEBGPU_BACKEND,
        None => DEFAULT_WEBGPU_BACKEND,
    }
}

fn bool_env(value: bool) -> String {
    if value {
        "1".to_string()
    } else {
        "0".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guest_envs_preserve_explicit_overrides() -> Result<()> {
        let envs = vec![
            "WEBGPU_BACKEND=metal".to_string(),
            "WEBGPU_ENABLED=0".to_string(),
            "WGPU_BACKEND=metal".to_string(),
        ];

        let middleware = WebGpuMiddleware::new(&envs)?;
        let merged = middleware.guest_envs(&envs);

        assert!(merged.iter().any(|env| env == "WEBGPU_BACKEND=metal"));
        assert!(merged.iter().any(|env| env == "WEBGPU_ENABLED=0"));
        assert_eq!(
            merged
                .iter()
                .filter(|env| env.starts_with("WEBGPU_BACKEND="))
                .count(),
            1
        );
        assert_eq!(
            merged
                .iter()
                .filter(|env| env.starts_with("WGPU_BACKEND="))
                .count(),
            1
        );

        Ok(())
    }

    #[test]
    fn guest_envs_fill_defaults() -> Result<()> {
        let envs = Vec::<String>::new();
        let middleware = WebGpuMiddleware::new(&envs)?;
        let merged = middleware.guest_envs(&envs);

        assert!(merged.iter().any(|env| env == "WEBGPU_ENABLED=1"));
        assert!(merged.iter().any(|env| env.starts_with("WEBGPU_BACKEND=")));
        assert!(
            merged
                .iter()
                .any(|env| env.starts_with("WEBGPU_MAX_BUFFER_SIZE="))
        );
        assert!(merged.iter().any(|env| env.starts_with("WGPU_BACKEND=")));

        Ok(())
    }

    #[test]
    fn guest_envs_strip_host_only_values() -> Result<()> {
        let envs = vec![
            "WEBGPU_DEVICE_PATH=/dev/dri/renderD128".to_string(),
            "WGPU_DEVICE_PATH=/dev/dri/renderD128".to_string(),
            format!("{}=broker-name", crate::broker::BROKER_ADDR_ENV),
        ];

        let middleware = WebGpuMiddleware::new(&envs)?;
        let merged = middleware.guest_envs(&envs);

        assert!(
            merged
                .iter()
                .all(|env| !env.starts_with("WEBGPU_DEVICE_PATH="))
        );
        assert!(
            merged
                .iter()
                .all(|env| !env.starts_with("WGPU_DEVICE_PATH="))
        );
        assert!(
            merged
                .iter()
                .all(|env| !env.starts_with(&format!("{}=", crate::broker::BROKER_ADDR_ENV)))
        );

        Ok(())
    }

    #[test]
    fn required_webgpu_only_needs_webgpu_enabled() {
        let envs = vec![
            "WEBGPU_REQUIRED=1".to_string(),
            "WEBGPU_ENABLED=1".to_string(),
        ];

        let middleware = WebGpuMiddleware::new(&envs).unwrap();
        assert!(middleware.config().required);
    }
}
