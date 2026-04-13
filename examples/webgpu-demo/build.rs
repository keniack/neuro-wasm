#[cfg(feature = "oci-v1-tar")]
use {
    anyhow::Context,
    oci_spec::image::{self as spec, Arch},
    oci_tar_builder::Builder,
    sha256::try_digest,
    std::env,
    std::fs::File,
    std::path::PathBuf,
};

#[cfg(not(feature = "oci-v1-tar"))]
fn main() {}

#[cfg(feature = "oci-v1-tar")]
fn main() {
    env_logger::init();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let output_dir = out_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    let app_path = output_dir.join("webgpu-demo.wasm");
    let layer_path = out_dir.join("layer.tar");
    tar::Builder::new(File::create(&layer_path).unwrap())
        .append_path_with_name(&app_path, "webgpu-demo.wasm")
        .unwrap();

    let mut builder = Builder::default();
    builder.add_layer(&layer_path);

    let config = spec::ConfigBuilder::default()
        .entrypoint(vec!["./webgpu-demo.wasm".to_owned()])
        .build()
        .unwrap();

    let layer_digest = try_digest(layer_path.as_path()).unwrap();
    let img = spec::ImageConfigurationBuilder::default()
        .config(config)
        .os("wasip1")
        .architecture(Arch::Wasm)
        .rootfs(
            spec::RootFsBuilder::default()
                .diff_ids(vec!["sha256:".to_owned() + &layer_digest])
                .build()
                .unwrap(),
        )
        .build()
        .context("failed to build image configuration")
        .unwrap();

    builder.add_config(
        img,
        "docker.io/keniack/webgpu-demo:local".to_string(),
    );

    let image_tar = out_dir.join("webgpu-demo-img.tar");
    let file = File::create(&image_tar).unwrap();
    builder.build(file).unwrap();
    std::fs::rename(&image_tar, output_dir.join("webgpu-demo-img.tar")).unwrap();
}
