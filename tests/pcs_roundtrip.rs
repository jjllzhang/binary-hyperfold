use binary_hyperfold::{
    BF128,
    parameters::{HyperFoldConfig, HyperFoldParams, MatrixKind},
    pcs::BinaryHyperFoldPcs,
    sumcheck,
};
use p3_binary_field::TowerLevel;
use p3_field::PrimeCharacteristicRing;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
#[test]
fn single_and_batched_roundtrip() {
    let mut rng = ChaCha8Rng::seed_from_u64(6);
    for r in [2usize, 4, 8] {
        for kind in [MatrixKind::Specialized, MatrixKind::General] {
            let p = HyperFoldParams {
                mu: 3 + 2 * r.ilog2() as usize,
                mu_0: Some(3),
                r,
                matrix_kind: kind,
                ..Default::default()
            };
            let c = HyperFoldConfig::derive(&p).unwrap();
            let pcs = BinaryHyperFoldPcs::setup(c, [11; 32]);
            let m: Vec<_> = (0..1 << p.mu)
                .map(|_| BF128::from_repr(rng.random()))
                .collect();
            let points: Vec<Vec<_>> = (0..2)
                .map(|_| (0..p.mu).map(|_| BF128::from_repr(rng.random())).collect())
                .collect();
            let values: Vec<_> = points.iter().map(|z| sumcheck::evaluate(&m, z)).collect();
            let (cm, data) = pcs.commit(m);
            for count in [1, 2] {
                let proof = pcs.open_batched(&data, &points[..count], &values[..count]);
                pcs.verify_batched(&cm, &points[..count], &values[..count], &proof)
                    .unwrap();
                let q = proof.queries.len();
                let d = pcs.config.d;
                let n = &pcs.config.n;
                let fields =
                    3 * (p.mu - pcs.config.mu_0) + pcs.config.k[0] + q * (r + (d - 1) * (r - 1));
                let hashes = d - 1
                    + proof
                        .paths
                        .iter()
                        .map(|p| p.sibling_hashes.len())
                        .sum::<usize>();
                assert_eq!(proof.size_in_bytes(), 16 * fields + 32 * hashes);
                let unpruned = q * n[..d].iter().map(|n| n.ilog2() as usize).sum::<usize>();
                assert!(hashes - (d - 1) <= unpruned);
                if q > 2 {
                    assert!(hashes - (d - 1) < unpruned);
                }
                assert_eq!(
                    proof.wire_size_in_bytes(),
                    bincode::serialize(&proof).unwrap().len()
                );
                let decoded = bincode::deserialize(&bincode::serialize(&proof).unwrap()).unwrap();
                pcs.verify_batched(&cm, &points[..count], &values[..count], &decoded)
                    .unwrap();
                let mut bad = proof.clone();
                bad.paths.pop();
                assert!(
                    pcs.verify_batched(&cm, &points[..count], &values[..count], &bad)
                        .is_err()
                );
                let mut bad = proof.clone();
                if let Some(hash) = bad.paths[0].sibling_hashes.first_mut() {
                    hash[0] ^= 1;
                    assert!(
                        pcs.verify_batched(&cm, &points[..count], &values[..count], &bad)
                            .is_err()
                    );
                }
                let mut bad = proof.clone();
                bad.final_table[0] += BF128::from_repr(1);
                assert!(
                    pcs.verify_batched(&cm, &points[..count], &values[..count], &bad)
                        .is_err()
                );
            }
        }
    }
}
#[test]
fn sumcheck_matches_naive() {
    use binary_hyperfold::sumcheck::prover::{
        fold_table_with_scratch, polynomial, polynomial_with_sum,
    };
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    for mu in [0, 3, 13] {
        let f: Vec<_> = (0..1usize << mu)
            .map(|_| BF128::from_repr(rng.random()))
            .collect();
        let w: Vec<_> = (0..f.len())
            .map(|_| BF128::from_repr(rng.random()))
            .collect();
        let point: Vec<_> = (0..mu).map(|_| BF128::from_repr(rng.random())).collect();
        assert_eq!(
            sumcheck::evaluate(&f, &point),
            f.iter()
                .zip(sumcheck::eq_table(&point))
                .map(|(&a, b)| a * b)
                .sum()
        );
        let sum = f.iter().zip(&w).map(|(&a, &b)| a * b).sum();
        let mut scratch = Vec::new();
        for bit in 0..mu {
            let p = polynomial(&f, &w, bit);
            assert_eq!(polynomial_with_sum(&f, &w, bit, sum).0, p.0);
            for alpha in [BF128::ZERO, BF128::ONE, BF128::from_repr(rng.random())] {
                let (mut a, mut b) = (f.clone(), w.clone());
                fold_table_with_scratch(&mut a, bit, alpha, &mut scratch);
                fold_table_with_scratch(&mut b, bit, alpha, &mut scratch);
                let stride = 1 << bit;
                let expected: Vec<_> = f
                    .chunks_exact(2 * stride)
                    .flat_map(|v| (0..stride).map(move |j| v[j] + alpha * (v[j] + v[j + stride])))
                    .collect();
                assert_eq!(a, expected);
                assert_eq!(
                    p.evaluate(alpha),
                    a.iter().zip(b).map(|(&a, b)| a * b).sum()
                );
            }
        }
    }
}

#[test]
fn primal_dual_subfield_codes_roundtrip() {
    use binary_hyperfold::parameters::DistanceMode;
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    for w in [16usize, 32, 64, 128] {
        for r in [2usize, 4, 8] {
            let p = HyperFoldParams {
                mu: 4 + 2 * r.ilog2() as usize,
                r,
                base_field_bits: w,
                distance_mode: DistanceMode::PrimalDual,
                ..Default::default()
            };
            let pcs = BinaryHyperFoldPcs::setup(HyperFoldConfig::derive(&p).unwrap(), [9; 32]);
            // The top input lies in the encoding subfield; challenges and
            // folded messages still use its BF128 scalar extension.
            let mask = u128::MAX >> (128 - w);
            let m: Vec<_> = (0..1 << p.mu)
                .map(|_| BF128::from_repr(rng.random::<u128>() & mask))
                .collect();
            let encoded = pcs.code.encode(pcs.config.d, &m);
            assert!(encoded.values.iter().all(|x| x.to_repr() & !mask == 0));
            let point: Vec<_> = (0..p.mu).map(|_| BF128::from_repr(rng.random())).collect();
            let value = sumcheck::evaluate(&m, &point);
            let (cm, data) = pcs.commit(m);
            let proof = pcs.open(&data, point.clone(), value);
            pcs.verify(&cm, point, value, &proof).unwrap();
        }
    }
}
