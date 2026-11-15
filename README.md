# Binary HyperFold

Research prototype under staged implementation. Mathematical specification `Binary_HyperFold.md`
is supplied privately during development and is deliberately ignored, never tracked.

## Current stage

Phase 1 implements the BF128 analytic distance and floating-point parameter search, plus an
independent exact-rational check of the specification's section 9.2 example. The fine primal/dual
evaluator is not yet available and returns an explicit unsupported result; it is not silently
replaced by a looser bound. Small coding fields, Flock and the PCS are not completed at this stage.

Dependencies are fetched from pinned remote revisions; there are no external path dependencies.
`BF128` is Plonky3's tower `BinaryField128`, not the differently represented `Ghash128`.

```
cargo check
cargo test --release --no-default-features --features parallel
python3 scripts/derive_params.py
```

Tests validate selected mathematical identities; they are not a security proof. Floating-point
search outputs alone are not certified numerical error bounds. No benchmark result is claimed yet.
