use std::fs::File;
use std::io::Read;

use anyhow::{Context, Result};
use context::{RuntimeContext, Source};
use path::PathResolve as _;

pub mod context;
pub(crate) mod path;

#[trait_variant::make(Send)]
pub trait Sandbox: Default + 'static {
    /// Run a WebAssembly container
    async fn run_wasi(&self, ctx: &impl RuntimeContext) -> Result<i32>;

    /// Check that the runtime can run the container.
    /// This checks runs after the container creation and before the container starts.
    /// By default it checks that the wasi_entrypoint is either:
    /// * a OCI image with wasm layers
    /// * a file with the `wasm` filetype header
    /// * a parsable `wat` file.
    async fn can_handle(&self, ctx: &impl RuntimeContext) -> Result<()> {
        // this async block is required to make the rewrite of trait_variant happy
        async move {
            let entrypoint = ctx.entrypoint();
            let source = entrypoint.source;

            let path = match source {
                Source::File(path) => path,
                Source::Oci(layers) => {
                    log::debug!(
                        "wasm can_handle accepted OCI wasm layers: layers={}, arg0={:?}, func={}, module={:?}, args={:?}",
                        layers.len(),
                        entrypoint.arg0,
                        entrypoint.func,
                        entrypoint.name,
                        ctx.args()
                    );
                    return Ok(());
                }
            };

            let candidates = path.resolve_in_path_or_cwd().collect::<Vec<_>>();
            log::debug!(
                "wasm can_handle probing file entrypoint: raw_path={}, arg0={:?}, func={}, module={:?}, args={:?}, resolved_candidates={:?}",
                path.display(),
                entrypoint.arg0,
                entrypoint.func,
                entrypoint.name,
                ctx.args(),
                candidates
            );

            let resolved = candidates.first().cloned().with_context(|| {
                format!(
                    "module not found for entrypoint {}; cwd={:?}; PATH={:?}",
                    path.display(),
                    std::env::current_dir().ok(),
                    std::env::var_os("PATH")
                )
            })?;

            let mut buffer = [0; 4];
            File::open(&resolved)?.read_exact(&mut buffer)?;

            if buffer.as_slice() != b"\0asm" {
                // Check if this is a `.wat` file
                log::debug!(
                    "wasm can_handle file header for {} is {:02x?}; attempting WAT parse",
                    resolved.display(),
                    buffer
                );
                wat::parse_file(&resolved).with_context(|| {
                    format!(
                        "entrypoint {} is neither a wasm binary nor a valid WAT file",
                        resolved.display()
                    )
                })?;
            } else {
                log::debug!(
                    "wasm can_handle recognized wasm binary entrypoint at {}",
                    resolved.display()
                );
            }

            Ok(())
        }
    }
}
