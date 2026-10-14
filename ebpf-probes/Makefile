# AegisCPG -- eBPF LSM enforcer build.
#
# Author: Pratham Kapoor <prathamkapoor027@gmail.com>
# https://github.com/PrathamKapoor
#
# WHAT:   Compiles lsm_enforcer.bpf.c into a loadable BPF object.
# WHY:    eBPF LSM needs kernel >= 5.7, CONFIG_BPF_LSM=y and a kernel booted with
#         `lsm=...,bpf`. A Windows dev host has none of that, so this Makefile is
#         intended to be invoked inside the Linux container defined by
#         Dockerfile.ebpf-runner (or WSL2), never directly on Windows.
# HOW:    `make` uses whatever clang with the BPF backend is on PATH. When none
#         is present -- the normal case on Windows -- use
#           docker build -f Dockerfile.ebpf-runner -t aegis-ebpf .
#         which reproduces the exact toolchain this file expects.

CLANG ?= clang
LLVM_STRIP ?= llvm-strip
ARCH ?= x86_64

CFLAGS := -O2 -g -Wall -Werror \
          -target bpf \
          -D__TARGET_ARCH_$(ARCH) \
          -I/usr/include/$(shell uname -m)-linux-gnu

SRC := lsm_enforcer.bpf.c
OBJ := lsm_enforcer.bpf.o

.PHONY: all
all: $(OBJ)

$(OBJ): $(SRC)
	$(CLANG) $(CFLAGS) -c $< -o $@
	@# Strip debug info only when llvm-strip is available; the BPF loader
	@# rejects objects larger than the 1 MiB kernel limit on some 5.x kernels.
	@command -v $(LLVM_STRIP) >/dev/null 2>&1 && $(LLVM_STRIP) -g $@ || true
	@echo "built $(OBJ):"
	@ls -l $(OBJ)

.PHONY: load
load: $(OBJ)
	@# Requires root and a kernel with CONFIG_BPF_LSM, booted with lsm=...,bpf.
	@bpftool prog loadall $(OBJ) /sys/fs/bpf/aegis
	@echo "loaded into /sys/fs/bpf/aegis"

.PHONY: verify
verify: $(OBJ)
	@# Confirm the verifier accepts every program before it ever reaches a kernel.
	@bpftool prog loadall $(OBJ) /sys/fs/bpf/aegis-verify && \
	  bpftool prog detach /sys/fs/bpf/aegis-verify && \
	  echo "verifier accepted all programs"

.PHONY: clean
clean:
	rm -f $(OBJ)

.PHONY: check
check:
	@command -v $(CLANG) >/dev/null 2>&1 || { \
	  echo "no clang with the BPF backend on PATH."; \
	  echo "On Windows or without kernel headers, build with:"; \
	  echo "  docker build -f Dockerfile.ebpf-runner -t aegis-ebpf ."; \
	  echo "The Rust test suite (cargo test --workspace) does not require this;"; \
	  echo "it exercises the identical decision table in the userspace simulator."; \
	  exit 1; }
	@echo "clang: $$($(CLANG) --version | head -1)"
	@uname -m