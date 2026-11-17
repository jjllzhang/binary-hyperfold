use binary_hyperfold::{
    BF16, BF32, BF128,
    code::{
        HyperFoldableCode, base_code,
        native::{NativeCode, NativeField, apply_h, apply_h128},
    },
    parameters::{DistanceMode, HyperFoldConfig, HyperFoldParams},
    pcs::native::{NativePcs, NativeProof},
    sumcheck::{self, prover::*},
};
use p3_binary_field::{BinaryField4, TowerLevel};
use p3_field::ExtensionField;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
fn lift<B: NativeField>(v: &[B]) -> Vec<BF128>
where
    BF128: ExtensionField<B>,
{
    v.iter().map(|&x| BF128::from(x)).collect()
}
fn check<B: NativeField>()
where
    BF128: ExtensionField<B>,
{
    let mut rng = ChaCha8Rng::seed_from_u64(31);
    assert_eq!(std::mem::size_of::<B>(), B::BITS / 8);
    for _ in 0..32 {
        let x = B::from_bits(rng.random());
        for c in 0..16 {
            let scalar = BinaryField4::from_repr(c);
            assert_eq!(x.mul_f16(scalar), x * B::from_bits(c as u128));
        }
        assert_eq!(x.mul_omega(), x * B::from_bits(2));
        for r in [2, 4, 8] {
            let row: Vec<_> = (0..r).map(|_| B::from_bits(rng.random())).collect();
            let mut a = row.clone();
            let mut b = lift(&row);
            for inv in [false, true] {
                apply_h(&mut a, inv);
                apply_h128(&mut b, inv);
                assert_eq!(lift(&a), b);
            }
            assert_eq!(a, row);
        }
    }
    let f: Vec<_> = (0..256).map(|_| B::from_bits(rng.random())).collect();
    let ext = lift(&f);
    let w: Vec<_> = (0..256).map(|_| BF128::from_repr(rng.random())).collect();
    let point: Vec<_> = (0..8).map(|_| BF128::from_repr(rng.random())).collect();
    let sum = f.iter().zip(&w).map(|(&a, &b)| b * a).sum();
    for bit in 0..8 {
        let expected = polynomial(&ext, &w, bit).0;
        assert_eq!(polynomial_base(&f, &w, bit).0, expected);
        assert_eq!(polynomial_base_with_sum(&f, &w, bit, sum).0, expected);
        let mut wide = ext.clone();
        fold_table(&mut wide, bit, point[0]);
        assert_eq!(fold_table_base(&f, bit, point[0]), wide);
    }
    assert_eq!(
        sumcheck::evaluate_base(&f, &point),
        sumcheck::evaluate(&ext, &point)
    );
    for c in [2usize, 4] {
        let small = &f[..16];
        assert_eq!(
            lift(&base_code::encode_native(small, c.ilog2() as usize)),
            base_code::encode(&lift(small), c.ilog2() as usize)
        );
        // Arbitrary extension coefficients use the same base-field generator matrix.
        let coeff: Vec<_> = (0..16).map(|_| BF128::from_repr(rng.random())).collect();
        let mut direct = vec![BF128::from_repr(0); 16 * c];
        for (j, &a) in coeff.iter().enumerate() {
            let mut unit = vec![B::ZERO; 16];
            unit[j] = B::ONE;
            for (v, b) in direct
                .iter_mut()
                .zip(base_code::encode_native(&unit, c.ilog2() as usize))
            {
                *v += a * b;
            }
        }
        assert_eq!(direct, base_code::encode(&coeff, c.ilog2() as usize));
        for r in [2usize, 4, 8] {
            let config = HyperFoldConfig::derive(&HyperFoldParams {
                mu: 8,
                r,
                c,
                base_field_bits: B::BITS,
                distance_mode: DistanceMode::PrimalDual,
                ..Default::default()
            })
            .unwrap();
            let a = NativeCode::<B>::setup(&config, &mut ChaCha8Rng::seed_from_u64(32));
            let b = HyperFoldableCode::setup(&config, &mut ChaCha8Rng::seed_from_u64(32));
            for (a, b) in a.twists.iter().zip(&b.twists) {
                assert_eq!(lift(a), *b);
            }
            for (a, b) in a.inverse_twists.iter().zip(&b.inverse_twists) {
                assert_eq!(lift(a), *b);
            }
            let word = a.encode(config.d, &f);
            let wide = b.encode(config.d, &ext);
            assert_eq!(lift(&word.values), wide.values);
            let beta: Vec<_> = (0..r).map(|_| BF128::from_repr(rng.random())).collect();
            assert_eq!(
                a.fold_top(config.d, &word, &beta).values,
                b.fold(config.d, &wide, &beta).values
            );
            let pcs = NativePcs::<B>::setup(config, [33; 32]);
            let value = sumcheck::evaluate_base(&f, &point);
            let (cm, data) = pcs.commit(f.clone());
            let proof = pcs.open(&data, point.clone(), value);
            pcs.verify(&cm, point.clone(), value, &proof).unwrap();
            let q = pcs.config.queries;
            let d = pcs.config.d;
            let fields = 3 * (8 - pcs.config.mu_0) + pcs.config.k[0] + q * (d - 1) * (r - 1);
            let hashes = d - 1
                + proof
                    .paths
                    .iter()
                    .map(|p| p.sibling_hashes.len())
                    .sum::<usize>();
            assert_eq!(
                proof.size_in_bytes(),
                16 * fields + (B::BITS / 8) * q * r + 32 * hashes
            );
            let bytes = bincode::serialize(&proof).unwrap();
            assert_eq!(proof.wire_size_in_bytes(), bytes.len());
            let restored: NativeProof<B> = bincode::deserialize(&bytes).unwrap();
            pcs.verify(&cm, point.clone(), value, &restored).unwrap();
            let mut bad = proof.clone();
            bad.paths.pop();
            assert!(pcs.verify(&cm, point.clone(), value, &bad).is_err());
            if !proof.paths[0].sibling_hashes.is_empty() {
                let mut bad = proof.clone();
                bad.paths[0].sibling_hashes[0][0] ^= 1;
                assert!(pcs.verify(&cm, point.clone(), value, &bad).is_err());
            }
        }
    }
}
#[test]
fn native_bf16_math_and_honest_flow() {
    check::<BF16>();
}
#[test]
fn native_bf32_math_and_honest_flow() {
    check::<BF32>();
}
