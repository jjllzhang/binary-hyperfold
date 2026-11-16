use binary_hyperfold::{
    BF64, BF128,
    code::{
        HyperFoldableCode, base_code,
        native64::{Code64, apply_h64, apply_h128},
        subfield::*,
    },
    parameters::{DistanceMode, HyperFoldConfig, HyperFoldParams},
    pcs::BinaryHyperFoldPcs64,
    sumcheck::{self, prover::*},
};
use p3_binary_field::{BinaryField4, TowerLevel};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
fn lift(x: &[BF64]) -> Vec<BF128> {
    x.iter().map(|&x| BF128::from(x)).collect()
}
#[test]
fn constants_and_circuits_match_extension() {
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    for _ in 0..32 {
        let x = BF64::from_repr(rng.random());
        for c in 0..16 {
            let c = BinaryField4::from_repr(c);
            assert_eq!(
                mul_f16_const_64(x, c),
                x * BF64::from_repr(c.to_repr() as u64)
            );
        }
        assert_eq!(BF128::from(mul_omega_64(x)), mul_omega(BF128::from(x)));
        for r in [2, 4, 8] {
            let input: Vec<_> = (0..r).map(|_| BF64::from_repr(rng.random())).collect();
            let mut native = input.clone();
            let mut wide = lift(&input);
            for inv in [false, true] {
                apply_h64(&mut native, inv);
                apply_h128(&mut wide, inv);
                assert_eq!(lift(&native), wide);
            }
            assert_eq!(native, input);
        }
    }
}
#[test]
fn base_domain_and_recursive_code_match() {
    let mut rng = ChaCha8Rng::seed_from_u64(18);
    for c in [2usize, 4] {
        for mu0 in [4usize, 6] {
            let m: Vec<_> = (0..1 << mu0)
                .map(|_| BF64::from_repr(rng.random()))
                .collect();
            assert_eq!(
                lift(&base_code::encode_64(&m, c.ilog2() as usize)),
                base_code::encode(&lift(&m), c.ilog2() as usize)
            );
        }
        for r in [2usize, 4, 8] {
            let config = HyperFoldConfig::derive(&HyperFoldParams {
                mu: 4 + 2 * r.ilog2() as usize,
                r,
                c,
                base_field_bits: 64,
                distance_mode: DistanceMode::PrimalDual,
                ..Default::default()
            })
            .unwrap();
            let native = Code64::setup(&config, &mut ChaCha8Rng::seed_from_u64(19));
            let wide = HyperFoldableCode::setup(&config, &mut ChaCha8Rng::seed_from_u64(19));
            for (a, b) in native.twists.iter().zip(&wide.twists) {
                assert_eq!(lift(a), *b);
            }
            for (a, b) in native.inverse_twists.iter().zip(&wide.inverse_twists) {
                assert_eq!(lift(a), *b);
            }
            let m: Vec<_> = (0..1 << config.params.mu)
                .map(|_| BF64::from_repr(rng.random()))
                .collect();
            let a = native.encode(config.d, &m);
            let b = wide.encode(config.d, &lift(&m));
            assert_eq!(lift(&a.values), b.values);
            let beta: Vec<_> = (0..r).map(|_| BF128::from_repr(rng.random())).collect();
            assert_eq!(
                native.fold_top(config.d, &a, &beta).values,
                wide.fold(config.d, &b, &beta).values
            );
        }
    }
}
#[test]
fn mixed_sumcheck_matches_lift() {
    let mut rng = ChaCha8Rng::seed_from_u64(20);
    for mu in [4, 8] {
        let f: Vec<_> = (0..1 << mu)
            .map(|_| BF64::from_repr(rng.random()))
            .collect();
        let wide = lift(&f);
        let w: Vec<_> = (0..f.len())
            .map(|_| BF128::from_repr(rng.random()))
            .collect();
        let point: Vec<_> = (0..mu).map(|_| BF128::from_repr(rng.random())).collect();
        assert_eq!(
            sumcheck::evaluate_base(&f, &point),
            sumcheck::evaluate(&wide, &point)
        );
        for bit in 0..mu {
            assert_eq!(polynomial_base(&f, &w, bit).0, polynomial(&wide, &w, bit).0);
            let mut folded = wide.clone();
            fold_table(&mut folded, bit, point[0]);
            assert_eq!(fold_table_base(&f, bit, point[0]), folded);
        }
    }
}
#[test]
fn native_pcs_happy_path_and_size() {
    let mut rng = ChaCha8Rng::seed_from_u64(21);
    for r in [2usize, 4, 8] {
        for c in [2usize, 4] {
            for mu in [12, 4 + r.ilog2() as usize] {
                let config = HyperFoldConfig::derive(&HyperFoldParams {
                    mu,
                    r,
                    c,
                    base_field_bits: 64,
                    distance_mode: DistanceMode::PrimalDual,
                    ..Default::default()
                })
                .unwrap();
                let pcs = BinaryHyperFoldPcs64::setup(config, [7; 32]);
                let message: Vec<_> = (0..1 << mu)
                    .map(|_| BF64::from_repr(rng.random()))
                    .collect();
                let point: Vec<_> = (0..mu).map(|_| BF128::from_repr(rng.random())).collect();
                let value = sumcheck::evaluate_base(&message, &point);
                let (cm, data) = pcs.commit(message);
                let proof = pcs.open(&data, point.clone(), value);
                pcs.verify(&cm, point, value, &proof).unwrap();
                let q = pcs.config.queries;
                let d = pcs.config.d;
                let fields = 3 * (mu - pcs.config.mu_0) + pcs.config.k[0] + q * (d - 1) * (r - 1);
                let hashes = d - 1
                    + q * pcs.config.n[..d]
                        .iter()
                        .map(|n| n.ilog2() as usize)
                        .sum::<usize>();
                assert_eq!(proof.size_in_bytes(), 16 * fields + 8 * q * r + 32 * hashes);
                assert_eq!(
                    proof.wire_size_in_bytes(),
                    bincode::serialize(&proof).unwrap().len()
                );
                assert!(cm.y0 == data.commitment.y0);
            }
        }
    }
}
