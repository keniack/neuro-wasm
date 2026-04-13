# syntax=docker/dockerfile:1.7

FROM rust:1.86-bookworm AS builder

WORKDIR /workspace
RUN rustup target add wasm32-wasip1

COPY . .

RUN cargo build --release --target wasm32-wasip1 -p webgpu-demo

FROM scratch

COPY --from=builder /workspace/target/wasm32-wasip1/release/webgpu-demo.wasm /webgpu-demo.wasm

ENTRYPOINT ["/webgpu-demo.wasm"]
CMD ["dispatch", "16"]
