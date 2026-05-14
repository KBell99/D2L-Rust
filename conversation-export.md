# Conversation Export — 2026-05-13

## Problem

Running a tch-rs program that creates a CUDA tensor panicked with:

```
called `Result::unwrap()` on an `Err` value: Torch("Could not run 'aten::empty.memory_format'
with arguments from the 'CUDA' backend. This could be because the operator doesn't exist for
this backend, or was omitted during the selective/custom build process (if using custom build).
```

### Code (`src/main.rs`)

```rust
use tch::{Device, Kind, Tensor};

fn main() {
    let t = Tensor::arange(12, (Kind::Float, Device::Cuda(0)));
    t.print();
}
```

### Environment

- GPU: NVIDIA RTX 3070
- Driver: 595.58.03, CUDA 13.2
- libtorch: `/usr/local/libtorch` → version `2.11.0+cu130`
- tch crate: `0.24.0`
- Linker: lld (via `rustup` toolchain)

---

## Root Cause

Linux's linker runs with `--as-needed` by default. This drops shared libraries from the final
binary if no compiled code directly references any of their symbols.

`libtorch_cuda.so` and `libtorch.so` register CUDA dispatch kernels via **C++ static
initializers** — no Rust (or tch-rs wrapper) code ever calls a symbol from them directly.
So the linker excluded both libraries from the binary even though `torch-sys`'s build script
correctly emitted `cargo:rustc-link-lib=torch_cuda`.

Without `libtorch_cuda.so` loaded at startup, the CUDA backend was never registered in
PyTorch's dispatcher, causing every CUDA tensor op to fail with the above error.

This was confirmed by:
- `ldd target/debug/d2l-rust | grep torch` — only `libtorch_cpu.so` appeared, not `libtorch_cuda.so`
- `LD_PRELOAD=/usr/local/libtorch/lib/libtorch_cuda.so ./target/debug/d2l-rust` — worked correctly

The `torch-sys` build script used to have a "dummy symbol" workaround for this but removed it
in recent versions with the comment "it seems that the dummy dependency is not necessary
anymore" — apparently a regression with lld.

---

## Fix

Created `build.rs` in the project root:

```rust
fn main() {
    println!("cargo:rustc-link-arg-bins=-Wl,--push-state,--no-as-needed,-ltorch_cuda,-ltorch,--pop-state");
    println!("cargo:rustc-link-search=native=/usr/local/libtorch/lib");
}
```

**Why this works:**

- `rustc-link-arg-bins` applies the linker flag only to binary targets (not build scripts or
  proc-macros), avoiding "library not found" errors in unrelated crates.
- `-Wl,--push-state,--no-as-needed,-ltorch_cuda,-ltorch,--pop-state` re-links both libraries
  at the end of the link command under `--no-as-needed`, forcing them into the binary
  regardless of direct symbol usage.
- The `rustc-link-search` line ensures the linker can find them even for our build script's
  link step.

After this fix, `ldd` shows `libtorch_cuda.so`, `libtorch.so`, `libc10_cuda.so`, and all
CUDA math libraries (cuBLAS, cuFFT, cuDNN, etc.) properly loaded.
