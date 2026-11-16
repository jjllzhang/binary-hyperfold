pub mod prover;
use crate::BF128;
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SumcheckPolynomial(pub [BF128; 3]);
impl SumcheckPolynomial {
    pub fn evaluate(self, t: BF128) -> BF128 {
        self.0[0] + t * (self.0[1] + t * self.0[2])
    }
    pub fn sum(self) -> BF128 {
        self.0[1] + self.0[2]
    }
}
pub fn eq(x: &[BF128], y: &[BF128]) -> BF128 {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y).map(|(&a, &b)| BF128::ONE + a + b).product()
}
pub fn eq_table(z: &[BF128]) -> Vec<BF128> {
    crate::code::folding::eq_weights(z)
}
pub fn evaluate(table: &[BF128], point: &[BF128]) -> BF128 {
    assert_eq!(table.len(), 1 << point.len());
    let mut v = table.to_vec();
    for &a in point {
        prover::fold_table(&mut v, 0, a);
    }
    v[0]
}

pub fn evaluate_base<B: p3_field::Field>(table: &[B], point: &[BF128]) -> BF128
where
    BF128: p3_field::ExtensionField<B>,
{
    assert_eq!(table.len(), 1 << point.len());
    if point.is_empty() {
        return BF128::from(table[0]);
    }
    let mut v = prover::fold_table_base(table, 0, point[0]);
    for &a in &point[1..] {
        prover::fold_table(&mut v, 0, a);
    }
    v[0]
}
