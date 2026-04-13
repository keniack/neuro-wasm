# syntax=docker/dockerfile:1.7

FROM docker.io/library/rust:1.86-slim-bookworm AS builder

RUN apt-get update && \
    apt-get install -y clang && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN rustup target add wasm32-wasip1

COPY . .

RUN cargo build --release --target wasm32-wasip1 -p image-classification-demo

FROM scratch

COPY --from=builder /workspace/target/wasm32-wasip1/release/image-classification-demo.wasm /image-classification-demo.wasm
COPY --from=builder /workspace/examples/image-classification-demo/images /images
COPY --from=builder /workspace/examples/image-classification-demo/models /models

ENTRYPOINT ["/image-classification-demo.wasm"]
CMD ["/images/red-apple.ppm", "/models/resnet50-demo.json"]
