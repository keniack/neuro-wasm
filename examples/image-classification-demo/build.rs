#[cfg(feature = "oci-v1-tar")]
use {
    anyhow::Context,
    oci_spec::image::{self as spec, Arch},
    oci_tar_builder::Builder,
    sha256::try_digest,
    std::env,
    std::fs::File,
    std::path::{Path, PathBuf},
};

#[cfg(not(feature = "oci-v1-tar"))]
fn main() {}

#[cfg(feature = "oci-v1-tar")]
fn main() {
    env_logger::init();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("models").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("images").display()
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let output_dir = out_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    let app_path = output_dir.join("image-classification-demo.wasm");
    let layer_path = out_dir.join("layer.tar");
    let mut layer = tar::Builder::new(File::create(&layer_path).unwrap());
    layer
        .append_path_with_name(&app_path, "image-classification-demo.wasm")
        .unwrap();

    for (source, destination) in [
        ("models/resnet50-demo.json", "models/resnet50-demo.json"),
        ("models/yolo-demo.json", "models/yolo-demo.json"),
        ("images/red-apple.ppm", "images/red-apple.ppm"),
        ("images/ocean.ppm", "images/ocean.ppm"),
        ("images/golden-retriever.ppm", "images/golden-retriever.ppm"),
    ] {
        append_file(&mut layer, manifest_dir.join(source), destination);
    }
    layer.finish().unwrap();

    let mut builder = Builder::default();
    builder.add_layer(&layer_path);

    let config = spec::ConfigBuilder::default()
        .entrypoint(vec![
            "./image-classification-demo.wasm".to_owned(),
            "/images/red-apple.ppm".to_owned(),
            "/models/resnet50-demo.json".to_owned(),
        ])
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
        "docker.io/keniack/image-classification-demo:local".to_string(),
    );

    let image_tar = out_dir.join("image-classification-demo-img.tar");
    let file = File::create(&image_tar).unwrap();
    builder.build(file).unwrap();
    std::fs::rename(
        &image_tar,
        output_dir.join("image-classification-demo-img.tar"),
    )
    .unwrap();
}

#[cfg(feature = "oci-v1-tar")]
fn append_file(layer: &mut tar::Builder<File>, source: PathBuf, destination: &str) {
    layer
        .append_path_with_name(&source, Path::new(destination))
        .unwrap();
}
