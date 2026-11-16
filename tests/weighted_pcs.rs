use binary_hyperfold::{
    BF128,
    parameters::{HyperFoldConfig, HyperFoldParams},
    pcs::{
        BinaryHyperFoldPcs,
        weighted::{TransparentWeight, WeightedClaim},
    },
    sumcheck,
    transcript::ProverTranscript,
};
use p3_binary_field::TowerLevel;
use p3_field::PrimeCharacteristicRing;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

struct DenseWeight {
    table: Vec<BF128>,
    label: u8,
}
impl TransparentWeight for DenseWeight {
    fn table(&self) -> Option<&[BF128]> {
        Some(&self.table)
    }
    fn evaluate(&self, point: &[BF128]) -> BF128 {
        sumcheck::evaluate(&self.table, point)
    }
    fn observe(&self, tr: &mut ProverTranscript) {
        tr.observe_bytes(&[self.label]);
        tr.observe_fields(&self.table);
    }
}

#[test]
fn weighted_roundtrip_and_tampering() {
    let mut rng = ChaCha8Rng::seed_from_u64(57);
    for r in [2usize, 4, 8] {
        let mu = 4 + 2 * r.ilog2() as usize;
        let pcs = BinaryHyperFoldPcs::setup(
            HyperFoldConfig::derive(&HyperFoldParams {
                mu,
                mu_0: Some(4),
                r,
                ..Default::default()
            })
            .unwrap(),
            [37; 32],
        );
        let message: Vec<_> = (0..1 << mu)
            .map(|_| BF128::from_repr(rng.random()))
            .collect();
        let weights: Vec<_> = (0..2)
            .map(|label| DenseWeight {
                table: (0..message.len())
                    .map(|_| BF128::from_repr(rng.random()))
                    .collect(),
                label,
            })
            .collect();
        let values: Vec<_> = weights
            .iter()
            .map(|w| message.iter().zip(&w.table).map(|(&a, &b)| a * b).sum())
            .collect();
        let claims: Vec<_> = weights
            .iter()
            .zip(&values)
            .map(|(weight, &value)| WeightedClaim { weight, value })
            .collect();
        let (cm, data) = pcs.commit(message);
        let proof = pcs.open_weighted(&data, &claims);
        pcs.verify_weighted(&cm, &claims, &proof).unwrap();
        assert_eq!(proof.base_polynomials.len(), 4);
        assert!(proof.size_in_bytes() <= proof.wire_size_in_bytes());
        let mut tampered = proof.clone();
        tampered.base_polynomials[0].0[0] += BF128::ONE;
        assert!(pcs.verify_weighted(&cm, &claims, &tampered).is_err());
        let mut tampered = proof.clone();
        tampered.base_polynomials.pop();
        assert!(pcs.verify_weighted(&cm, &claims, &tampered).is_err());
        let mut tampered = proof.clone();
        tampered.queries[0][0].values[0] += BF128::ONE;
        assert!(pcs.verify_weighted(&cm, &claims, &tampered).is_err());
        let bad = [
            WeightedClaim {
                weight: &weights[0] as &dyn TransparentWeight,
                value: values[0] + BF128::ONE,
            },
            WeightedClaim {
                weight: &weights[1],
                value: values[1],
            },
        ];
        assert!(pcs.verify_weighted(&cm, &bad, &proof).is_err());
    }
}
