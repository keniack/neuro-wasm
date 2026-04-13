PREFIX ?= /usr/local
INSTALL ?= install
CARGO ?= cargo
OPT_PROFILE ?= debug
TARGET ?=
APT_GET ?= apt-get

DEBIAN_BUILD_DEPS = build-essential clang libclang-dev libc6-dev libseccomp-dev vulkan-tools libvulkan1

ifeq ($(OPT_PROFILE),release)
RELEASE_FLAG = --release
PROFILE_DIR = release
else
RELEASE_FLAG =
PROFILE_DIR = debug
endif

ifneq ($(TARGET),)
TARGET_FLAG = --target $(TARGET)
BIN_DIR = target/$(TARGET)/$(PROFILE_DIR)
else
TARGET_FLAG =
BIN_DIR = target/$(PROFILE_DIR)
endif

.PHONY: build
build: build-webgpu build-examples

.PHONY: install-build-deps-debian
install-build-deps-debian:
	sudo $(APT_GET) update
	sudo $(APT_GET) install -y $(DEBIAN_BUILD_DEPS)

.PHONY: build-webgpu
build-webgpu:
	$(CARGO) build $(TARGET_FLAG) $(RELEASE_FLAG) -p containerd-shim-webgpu

.PHONY: build-examples
build-examples:
	$(CARGO) build --target wasm32-wasip1 $(RELEASE_FLAG) -p webgpu-demo
	$(CARGO) build --target wasm32-wasip1 $(RELEASE_FLAG) -p image-classification-demo

.PHONY: build-examples-oci
build-examples-oci:
	$(CARGO) build --target wasm32-wasip1 $(RELEASE_FLAG) -p webgpu-demo --features oci-v1-tar
	$(CARGO) build --target wasm32-wasip1 $(RELEASE_FLAG) -p image-classification-demo --features oci-v1-tar

.PHONY: install-webgpu
install-webgpu:
	test -x $(BIN_DIR)/containerd-shim-webgpu-v1 || (echo "missing $(BIN_DIR)/containerd-shim-webgpu-v1; run 'make build-webgpu' first" && exit 1)
	mkdir -p $(PREFIX)/bin
	$(INSTALL) $(BIN_DIR)/containerd-shim-webgpu-v1 $(PREFIX)/bin/

.PHONY: fmt
fmt:
	$(CARGO) fmt --all
