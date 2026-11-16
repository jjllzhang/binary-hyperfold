# Binary HyperFold

An experimental implementation of Binary HyperFold, a multilinear polynomial commitment scheme over binary fields. It combines a recursively foldable code with sumcheck and Merkle commitments. The repository includes PCS experiments and a proof-of-concept R1CS integration using weighted openings.

**WARNING:** This is a research prototype for studying the protocol and measuring its performance. It has not been audited and is not intended for production use.

## Usage

Cargo builds in this repository default to `-C target-cpu=native` via `.cargo/config.toml`, including tests and benchmarks. Release builds also use thin LTO and a single codegen unit. Binaries built this way may not run on another CPU; for a portable build, override the repository setting with `RUSTFLAGS="-C target-cpu=generic"` when invoking Cargo.

Run the PCS tests and parameter checks:

```sh
cargo test --release --no-default-features --features parallel
python3 scripts/derive_params.py
```

Run a small PCS experiment:

```sh
cargo run --release --bin bench_pcs -- \
  --mu 10 --r 4 --c 2 --security-bits 100 \
  --threads 4 --warmup 0 --iters 1
```

The optional R1CS experiment compares weighted Binary HyperFold openings with Ligerito:

```sh
cargo test --release
cargo run --release --bin bench_flock -- \
  --hash blake3 --log2-batches 9 --r 4 \
  --threads 4 --warmup 0 --iters 1
```

Use `--help` on either benchmark binary for other configurations. Larger instances can require substantial memory and time. The R1CS experiment tests circuit satisfaction; it does not expose a separate public hash input or digest statement.
