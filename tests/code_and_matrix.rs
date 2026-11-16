use binary_hyperfold::{
    BF128,
    code::{
        matrix_h::{LocalMatrix, MatrixKind},
        subfield::*,
    },
};
use p3_binary_field::TowerLevel;
use p3_field::{Field, PrimeCharacteristicRing};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
#[test]
fn subfield_identities_and_scaling() {
    let z = zeta();
    assert_eq!(z.square().square() + z + BF128::ONE, BF128::ZERO);
    assert_eq!(z.square() + z, omega());
    assert_eq!(omega().square() + omega() + BF128::ONE, BF128::ZERO);
    let t = tau();
    assert_eq!(t.square() + t, z);
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..16 {
        let x = BF128::from_repr(rng.random());
        for c in 0..16 {
            assert_eq!(
                mul_f16_const(x, p3_binary_field::BinaryField4::from_repr(c)),
                x * f16(c)
            );
        }
        assert_eq!(mul_zeta_inv(mul_zeta(x)), x);
    }
}
#[test]
fn h_matrices_match_naive_and_inverse() {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    for r in [2usize, 4, 8] {
        for kind in [MatrixKind::Specialized, MatrixKind::General] {
            let h = LocalMatrix::new(r, kind);
            for _ in 0..8 {
                let x: Vec<_> = (0..r).map(|_| BF128::from_repr(rng.random())).collect();
                let expected: Vec<_> = (0..r)
                    .map(|i| (0..r).map(|j| h.dense[i * r + j] * x[j]).sum::<BF128>())
                    .collect();
                let mut y = x.clone();
                h.apply(&mut y, false);
                assert_eq!(y, expected);
                h.apply(&mut y, true);
                assert_eq!(y, x);
            }
            if r == 8 && kind == MatrixKind::Specialized {
                let v = [BF128::ONE, omega(), zeta()];
                for i in 0..8 {
                    for j in 0..8 {
                        let ij = i ^ j;
                        let x = (0..3)
                            .filter(|k| ij & (1 << k) != 0)
                            .map(|k| v[k])
                            .sum::<BF128>();
                        assert_eq!(h.dense[i * 8 + j], (tau() + x).inverse());
                    }
                }
            }
            for size in 1..=r {
                let masks: Vec<_> = (1usize..1 << r)
                    .filter(|v| v.count_ones() as usize == size)
                    .collect();
                for &im in &masks {
                    for &jm in &masks {
                        let mut a = vec![];
                        for i in 0..r {
                            if im & (1 << i) != 0 {
                                for j in 0..r {
                                    if jm & (1 << j) != 0 {
                                        a.push(h.dense[i * r + j]);
                                    }
                                }
                            }
                        }
                        binary_hyperfold::code::matrix_h::invert(&a, size);
                    }
                }
            }
        }
    }
}
#[test]
fn encoding_folding_commutes() {
    use binary_hyperfold::{
        code::{
            HyperFoldableCode,
            folding::{eq_weights, symbol},
        },
        parameters::{HyperFoldConfig, HyperFoldParams},
    };
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    for r in [2usize, 4, 8] {
        for kind in [MatrixKind::Specialized, MatrixKind::General] {
            let nu = r.ilog2() as usize;
            let p = HyperFoldParams {
                mu: 3 + 2 * nu,
                mu_0: Some(3),
                r,
                matrix_kind: kind,
                ..Default::default()
            };
            let c = HyperFoldConfig::derive(&p).unwrap();
            let code = HyperFoldableCode::setup(&c, &mut rng);
            let m: Vec<_> = (0..c.k[c.d])
                .map(|_| BF128::from_repr(rng.random()))
                .collect();
            let w = code.encode(c.d, &m);
            let alpha: Vec<_> = (0..nu).map(|_| BF128::from_repr(rng.random())).collect();
            let beta = eq_weights(&alpha);
            let k = c.k[c.d - 1];
            let mixed: Vec<_> = (0..k)
                .map(|j| (0..r).map(|a| beta[a] * m[a * k + j]).sum())
                .collect();
            let expected = code.encode(c.d - 1, &mixed);
            let got = code.fold(c.d, &w, &beta);
            assert_eq!(got.values, expected.values);
            for j in 0..got.values.len() {
                assert_eq!(symbol(&got, j), symbol(&expected, j));
            }
        }
    }
}
