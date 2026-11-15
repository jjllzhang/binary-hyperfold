use binary_hyperfold::{
    BF128,
    parameters::{HyperFoldConfig, HyperFoldParams, MatrixKind},
    pcs::BinaryHyperFoldPcs,
    sumcheck,
};
use p3_binary_field::TowerLevel;
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
                let hashes = d - 1 + q * n[..d].iter().map(|n| n.ilog2() as usize).sum::<usize>();
                assert_eq!(proof.size_in_bytes(), 16 * fields + 32 * hashes);
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
    use binary_hyperfold::sumcheck::prover::{fold_table, polynomial};
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    for n in [8, 16] {
        for bit in 0..(n as usize).ilog2() as usize {
            let f: Vec<_> = (0..n).map(|_| BF128::from_repr(rng.random())).collect();
            let w: Vec<_> = (0..n).map(|_| BF128::from_repr(rng.random())).collect();
            let p = polynomial(&f, &w, bit);
            let alpha = BF128::from_repr(rng.random());
            let (mut a, mut b) = (f, w);
            fold_table(&mut a, bit, alpha);
            fold_table(&mut b, bit, alpha);
            assert_eq!(
                p.evaluate(alpha),
                a.iter().zip(b).map(|(&a, b)| a * b).sum()
            );
        }
    }
}
