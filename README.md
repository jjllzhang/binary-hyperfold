# Binary HyperFold

Self-contained **research prototype under staged implementation**, based on the privately supplied
`Binary_HyperFold.md`. That specification is ignored and must never be staged or committed.

## Implemented core

- Plonky3 pinned at `0d803737b681774186cd283b1fd2d8e493655c82`, no external path dependencies.
- `BF128 = BinaryField128` in the tower basis (not GHASH raw bit representation).
- Integer-ceiling analytic code-distance schedule, floating-point MCA query search, exact-rational
  section 9.2 example checks. Auto base dimension is the smallest mu0 >= 4 with arity divisibility;
  the paper example explicitly uses mu0=8. PrimalDual is still **unsupported** pending its evaluator.
- Subfield bitplane maps, H2/H4 and unrolled H8, Cauchy baseline, additive RS, recursive encode/fold.
- Interleaved degree-2 sumcheck, BLAKE3 BinaryChallenger, Plonky3 MerkleTreeMmcs, single and multipoint
  openings. All claims are bound before gamma; m-point error includes mL/q, not L/q.
- Exact field/hash payload accounting and separately reported bincode wire size. Intermediate queried
  symbols are omitted and reconstructed before authenticating the corresponding Merkle row.

## Validation and PCS measurement

```
cargo check
cargo test --release --no-default-features --features parallel
python3 scripts/derive_params.py
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin bench_pcs -- \
  --mu 20 --r 2,4,8 --c 2 --security-bits 100 --threads 32
cargo bench --bench primitives
python3 scripts/benchmark_pcs.py --mu 20 --r 2,4,8 --c 2 --out /tmp/hyperfold-run
```

`proof_bytes` includes commitment root, OOD point/value and evaluation-proof field/hash payload;
`wire_proof_bytes` also includes serde container framing. Commit includes encoding, Merkle and OOD
sampling/evaluation as Protocol 4.1 requires. Setup, input generation/cloning and external claim
computation are excluded. Open includes its protocol weight tables and sumcheck, not duplicate
external evaluations in release. Verify checks all sumchecks, final scalar identity, and reconstructed
Merkle openings against the public statement. Tests are mathematical/functional checks, **not a
security proof**. The floating-point optimizer is not a rigorously rounded numerical certificate;
the supplied exact script certifies the given section 9.2 rational example only.

Public setup diagonals are generated from a caller-supplied public seed using ChaCha8. The note's
independent-uniform setup theorem is an ideal distribution statement, not an information-theoretic
claim about a finite deterministic PRNG seed. No benchmark campaign results are bundled or claimed.

## Not yet implemented

PrimalDual schedule generation, narrower base-field storage, and the end-to-end Flock adapter and
Flock benchmarks remain work in progress. Flock option B (separate ring-switch sumcheck, followed by
ordinary HyperFold opening) was selected; no placeholder Flock proof or timing is provided.
