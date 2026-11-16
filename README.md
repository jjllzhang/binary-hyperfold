# Binary HyperFold

Self-contained **research prototype**, based on the privately supplied
`Binary_HyperFold.md`. That specification is ignored and must never be staged or committed.

## Implemented core

- Plonky3 pinned at `0d803737b681774186cd283b1fd2d8e493655c82`, no external path dependencies.
- `BF128 = BinaryField128` in the tower basis (not GHASH raw bit representation).
- Integer-ceiling analytic code-distance schedule, floating-point MCA query search, exact-rational
  section 9.2 example checks. Auto base dimension is the smallest mu0 >= 4 with arity divisibility;
  the paper example explicitly uses mu0=8. PrimalDual searches a grouped B_fine upper envelope
  jointly for primal/dual thresholds (not an exact B_fine/global-minimum solver).
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
the included reference tool certifies exported schedules as described below.

Public setup diagonals are generated from a caller-supplied public seed using ChaCha8. The note's
independent-uniform setup theorem is an ideal distribution statement, not an information-theoretic
claim about a finite deterministic PRNG seed. No benchmark campaign results are bundled or claimed.

## Flock adapter (option B)

The optional `flock` feature is enabled by default. Dependencies are pinned to Flock
`dcd17533128a595e475b3dbba56c3dc2ad65161f`. The adapter uses a genuine field isomorphism
between GHASH and the tower (not a raw-bit reinterpretation). It runs Flock's Boolean zerocheck
and lincheck, ring-switches both resulting claims, and performs a separate linear-cost sumcheck
before an ordinary HyperFold opening. The Flock transcript fixes the public circuit/commitment
and reduction messages; the separate domain-separated HyperFold transcript binds the same
commitment and the resulting point/value. No Ligerito verification step is skipped or error ignored.

```
cargo test --release
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin bench_flock -- \
  --hash blake3 --log2-batches 10 --r 2,4,8 --threads 32
# Also supports --hash sha256.
```

The benchmark compares the same fast-generated row-major Flock witness/a/b/lincheck buffers.
Input clones for repeated proving, setup and R1CS construction are outside timing. `witness_ms`
records actual witness and auxiliary packing once and is added to every reported total. Commit
includes the GHASH→tower conversion. `ring_switch_ms` includes option B's extra standalone
sumcheck pass; Ligerito's corresponding work is fused into `open_ms` and shown as `fused-in-open`,
not a fabricated separate zero-time phase. Overall wall time includes preparation, transcript,
allocation and cleanup outside individual phase windows. Total proof KB is actual bincode framing,
not the pure PCS payload metric. These metrics are deliberately distinguished.

Ligerito uses its upstream Fast100 profile. HyperFold uses the 100-bit-target PCS schedule with
Flock's no-grinding PIOP path. This does not assert a complete composition security certificate or
identical security assumptions. The adapter exposes a circuit-satisfaction research R1CS proof,
not a separate public digest/input statement binding API. Upstream Flock may use helper threads
in addition to its configured Rayon pools; `--threads` fixes both Rayon pools, not every OS thread.
No formal campaign or exclusive-host claim is made by the supplied smoke tests.

## Parameter export and independent verification

```
cargo build --release --bin bench_pcs
python3 scripts/derive_params.py --generate --mu 20 --r 2,4,8 --c 2,4 \
  --distance-mode primal-dual --output /tmp/schedule.json
# Or: bench_pcs --iters 0 --schedule-json /tmp/schedule.json [other flags]
python3 scripts/derive_params.py --verify-json /tmp/schedule.json
cd scripts && python3 -m unittest -v test_fine_bound
```

The Rust log-space search proposes thresholds/queries; it is **not** directed arithmetic.
`derive_params.py` checks every primal and dual endpoint with the independent Decimal-interval
reference evaluator, then checks the distance, radius and full setup/field/query budget using
exact rational inequalities. For Analytic it checks integer ceilings and Bernoulli conditions
instead. The automatic benchmark runner performs this check before measuring each configuration.
The certificates exclude Fiat–Shamir, Merkle, deterministic public-setup PRNG and outer Flock errors.
The setup theorem is conditional on the ideal random code ensemble, not a seed security proof.

The reference files `fine_bound.py`, `binomial_interval.py`, `source-helpers/arithmetic.py` and
`test_fine_bound.py` were supplied by Theorist in the 2026-11-15 parameter audit. Method:
`fine_group_binomial_geometric_upper_with_coarse_min`; `exact_Bfine_minimum_claimed=False`.
The Rust port uses exact integer mode ratios, conservative mathematical envelopes with floating
logs, and rejects unsupported fast-field conditions. It currently limits fine search to mu<=32.
Run the independent verifier for **each** newly used configuration; matching the examples alone
is not a numerical certificate for other configurations.

`--base-field-bits 16|32|64|128` controls the encoding field (public twists and base-code domain);
in the default `bf128` path, all challenges, folded tables, storage and proof field elements remain
BF128. This is a scalar
extension of the base-field code, **not** a narrow-storage/arithmetic optimization. Top benchmark
inputs are sampled in the chosen subfield; reported bytes are actual 16-byte field payloads.
The analytic setup bound usually cannot meet 100 bits with a small base field. Fine mode may work,
for example `--mu 20 --mu0 8 --r 4 --c 2 --base-field-bits 32 --distance-mode primal-dual`.
The fast method has explicit large-field preconditions and can report unsupported configurations;
that is not a claim that no other bound or protocol parameter can work.

## Native BF64 path

`BinaryHyperFoldPcs64` and `bench_pcs --storage bf64 --base-field-bits 64` provide a
single-claim, specialized-H alternative to the unchanged default `--storage bf128` path.
The native path stores messages, top codewords, public twists and their inverses as BF64;
recursive encoding uses BF64 arithmetic. H coefficients are typed GF(16) values and both
64/128-bit circuits use only XOR and small-constant linear maps. The first sumcheck/fold binds
the base message straight into a half-length BF128 table; subsequent messages and oracles
remain BF128. Bottom verification uses the same Cantor-domain encoder over the extension.

The top Merkle leaves hash native 8-byte elements, rather than 16-byte embedded elements.
The native transcript has a separate domain tag. Roots/challenges are therefore not expected
to match the baseline, although codewords and public twists agree after the tower embedding.
Top query symbols use 8 bytes in the native proof payload, all other field symbols still use 16.
This format and first-binding allocation change are part of the compared variant, not an
isolated multiplication-kernel optimization or proof of a particular performance mechanism.
Actual bincode size remains separately reported; the two proof types have different framing.

```
RUSTFLAGS="-C target-cpu=native" cargo run --release --bin bench_pcs -- \
  --mu 24 --r 4 --c 2 --base-field-bits 64 --distance-mode primal-dual \
  --storage bf64 --threads 32
# Repeat with --storage bf128 for the embedded baseline, keeping all other flags identical.
```

Native BF64 support does not change the default Flock adapter, does not implement native BF32
or general/Cauchy matrices, and does not establish a new security proof. Every measured schedule
must still be exported and independently certified before it is used as a target-bit parameter.
No native-versus-embedded performance result is claimed by this README alone.
