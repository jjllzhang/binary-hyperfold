#![cfg(feature = "flock")]
use binary_hyperfold::{
    BF128,
    adapters::flock::*,
    parameters::{HyperFoldConfig, HyperFoldParams},
    pcs::BinaryHyperFoldPcs,
};
use p3_binary_field::TowerLevel;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
#[test]
fn field_bridge_matches_naive() {
    let mut rng = ChaCha8Rng::seed_from_u64(89);
    for _ in 0..32 {
        let a = BF128::from_repr(rng.random());
        let b = BF128::from_repr(rng.random());
        assert_eq!(from_flock(to_flock(a)), a);
        assert_eq!(from_flock(to_flock(a) * to_flock(b)), a * b);
    }
}
#[test]
fn blake3_r1cs_roundtrip() {
    use flock_prover::r1cs_hashes::blake3;
    let log = 3;
    let r = blake3::build_block_r1cs(log);
    let block = ([0u32; 8], [0u32; 16], 0, 64, 0);
    let inputs = vec![block; 1 << log];
    let (z, a, b, zlin) = blake3::generate_witness_with_ab_packed_and_lincheck(&inputs, log);
    let c = HyperFoldConfig::derive(&HyperFoldParams {
        mu: r.m - 7,
        r: 4,
        ..Default::default()
    })
    .unwrap();
    let pcs = BinaryHyperFoldPcs::setup(c, [61; 32]);
    let circuit = r.csc_lincheck_circuit();
    let (proof, _) = prove_r1cs_with_binary_hyperfold(&r, &pcs, z, a, b, &zlin, circuit);
    assert!(verify_r1cs_with_binary_hyperfold(&r, &pcs, &proof, circuit));
}
#[test]
fn sha256_r1cs_roundtrip() {
    use flock_prover::r1cs_hashes::sha2;
    let log = 3;
    let r = sha2::build_block_r1cs(log);
    let inputs = vec![([0u32; 8], [0u32; 16]); 1 << log];
    let (z, a, b, zlin) = sha2::generate_witness_with_ab_packed_and_lincheck(&inputs, log);
    let c = HyperFoldConfig::derive(&HyperFoldParams {
        mu: r.m - 7,
        r: 4,
        ..Default::default()
    })
    .unwrap();
    let pcs = BinaryHyperFoldPcs::setup(c, [62; 32]);
    let circuit = r.csc_lincheck_circuit();
    let (proof, _) = prove_r1cs_with_binary_hyperfold(&r, &pcs, z, a, b, &zlin, circuit);
    assert!(verify_r1cs_with_binary_hyperfold(&r, &pcs, &proof, circuit));
}
