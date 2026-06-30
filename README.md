# d2l-rust

A Rust implementation of the exercises from [Dive into Deep Learning (d2l.ai)](https://d2l.ai/), using [`tch-rs`](https://github.com/LaurentMazare/tch-rs) — the Rust bindings for PyTorch's `libtorch` — as the primary tensor computation backend.

Where the original book uses Python with PyTorch or MXNet, this repo translates each exercise into idiomatic Rust, adapting the API to `tch-rs` while keeping the mathematical intent identical.

## Crate dependencies

| Crate | Role |
| --- | --- |
| [`tch`](https://crates.io/crates/tch) | Tensor operations and automatic differentiation via `libtorch` |
| [`polars`](https://crates.io/crates/polars) | DataFrame loading and manipulation (replaces `pandas`) |
| [`ndarray`](https://crates.io/crates/ndarray) | N-dimensional arrays; bridge between `polars` and `tch` tensors |
| [`plotters`](https://crates.io/crates/plotters) | Chart and plot generation (replaces `matplotlib`) |
| [`rand`](https://crates.io/crates/rand) | Random sampling for probability exercises |
| [`statrs`](https://crates.io/crates/statrs) | Statistical distributions |

## Prerequisites

`tch-rs` requires a local `libtorch` installation. The `build.rs` script hard-codes the library search path to `/usr/local/libtorch/lib` and links `torch` and `torch_cuda`. Adjust `build.rs` if your install path differs.

```text
/usr/local/libtorch/
└── lib/
    ├── libtorch.so
    ├── libtorch_cuda.so
    └── ...
```

A CUDA-capable GPU is optional. Exercises use `Device::cuda_if_available()` where the book demonstrates GPU usage, so they fall back to CPU automatically.

## Running an exercise

Each chapter section is a standalone binary under `src/bin/`:

```bash
cargo run --bin ch2_1   # Chapter 2.1 — Data Manipulation
cargo run --bin ch2_2   # Chapter 2.2 — Data Preprocessing
cargo run --bin ch2_3   # Chapter 2.3 — Linear Algebra
cargo run --bin ch2_4   # Chapter 2.4 — Calculus
cargo run --bin ch2_5   # Chapter 2.5 — Automatic Differentiation
cargo run --bin ch2_6   # Chapter 2.6 — Probability and Statistics
```

Chapters that produce plots write PNG files to a `plots/` directory in the working directory. Create it before running those exercises:

```bash
mkdir -p plots
cargo run --bin ch2_4
cargo run --bin ch2_6
```

## Chapter coverage

### 2.1 — Data Manipulation ([src/bin/ch2_1.rs](src/bin/ch2_1.rs))

Covers the `tch-rs` equivalents of the book's NumPy/PyTorch tensor basics:

- Creating tensors with `Tensor::arange` and `Tensor::from_slice`
- Indexing and slicing with `.i()`
- In-place operations and memory identity via `data_ptr`
- Elementwise arithmetic, `Tensor::cat`, equality checks, and reduction with `.sum()`
- Converting tensors to Rust scalar values with `.double_value()`

### 2.2 — Data Preprocessing ([src/bin/ch2_2.rs](src/bin/ch2_2.rs))

Replaces `pandas` with `polars` for CSV ingestion and feature engineering:

- Writing and reading a CSV file with `CsvReadOptions`
- Splitting input features from targets via `DataFrame::select`
- One-hot encoding categorical columns with `columns_to_dummies`
- Imputing missing values with column means using the lazy API (`fill_null` + `mean`)
- Converting a `polars` `DataFrame` to an `ndarray` array and then to a `tch` tensor via the shared `ndarray_to_tensor` utility

### 2.3 — Linear Algebra ([src/bin/ch2_3.rs](src/bin/ch2_3.rs))

Walks through the book's linear algebra section using `tch-rs`:

- Scalars, vectors, matrices, and higher-rank tensors
- Transpose (`.tr()`), symmetry checks
- Hadamard (elementwise) product and scalar broadcasting
- Reduction: sum, mean over axes; non-reduction sum with `keepdim`
- Cumulative sum (`.cumsum()`)
- Dot products (`.dot()`), matrix–vector products (`.mv()`), and matrix–matrix multiplication (`.mm()`)
- L2 and L1 norms via `.norm()` and `.abs().sum()`

### 2.4 — Calculus ([src/bin/ch2_4.rs](src/bin/ch2_4.rs))

Implements numerical differentiation and visualization (no `tch` required here):

- Numerical limit approximation of the derivative of `f(x) = 3x² - 4x` at `x = 1`
- Plots `f(x)` and its tangent line at `x = 1` using `plotters`, saved to `plots/ch2_4.png`

### 2.5 — Automatic Differentiation ([src/bin/ch2_5.rs](src/bin/ch2_5.rs))

Demonstrates `tch-rs`'s autograd engine:

- Enabling gradient tracking with `.requires_grad_(true)`
- Forward pass and backward pass with `.backward()`
- Verifying gradients (e.g., `∇(2xᵀx) = 4x`)
- Resetting gradient buffers with `.zero_grad()`
- Detaching a subgraph from the computation graph with `.detach()` to treat intermediate values as constants

### 2.6 — Probability and Statistics ([src/bin/ch2_6.rs](src/bin/ch2_6.rs))

Covers probability fundamentals and the law of large numbers:

- Simulating coin tosses with `rand`
- Multinomial sampling (hand-rolled using `WeightedIndex` because `tch-rs` does not expose `torch.distributions.Multinomial`)
- Computing cumulative counts and running probability estimates as tensors
- Plotting convergence of empirical estimates to `P = 0.5` with `plotters`, saved to `plots/ch2_6_coin_estimates.png`

## Utility library ([src/utils.rs](src/utils.rs))

Shared helpers used across chapters:

| Function | Description |
| --- | --- |
| `ndarray_to_tensor` | Converts an `ndarray::Array<f64, D>` to a `tch::Tensor` |
| `tensor_to_ndarray` | Converts a `tch::Tensor` to an `ndarray::ArrayD<f64>` |
| `multinomial` | Draws multinomial samples into an `ndarray` array; accepts `()`, `usize`, or tuple size arguments via the `IntoSize` trait |

The `ndarray`↔`tch` conversion utilities exist because `polars` exports DataFrames to `ndarray` and `tch-rs` does not natively consume `polars` frames.

## Project structure

```text
d2l-rust/
├── build.rs              # libtorch linker flags
├── Cargo.toml
├── data/
│   └── house_tiny.csv    # Sample dataset for ch2_2
├── plots/                # Generated chart output (create manually)
└── src/
    ├── lib.rs
    ├── utils.rs          # Shared ndarray↔tensor utilities and multinomial sampler
    └── bin/
        ├── ch2_1.rs      # 2.1 Data Manipulation
        ├── ch2_2.rs      # 2.2 Data Preprocessing
        ├── ch2_3.rs      # 2.3 Linear Algebra
        ├── ch2_4.rs      # 2.4 Calculus
        ├── ch2_5.rs      # 2.5 Automatic Differentiation
        └── ch2_6.rs      # 2.6 Probability and Statistics
```
