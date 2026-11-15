use super::SumcheckPolynomial;
use crate::BF128;
use p3_field::PrimeCharacteristicRing;
use p3_maybe_rayon::prelude::*;
pub fn polynomial(f: &[BF128], w: &[BF128], bit: usize) -> SumcheckPolynomial {
    let stride = 1 << bit;
    let coeff = f
        .par_chunks(2 * stride)
        .zip(w.par_chunks(2 * stride))
        .map(|(fs, ws)| {
            let mut c = [BF128::ZERO; 3];
            for j in 0..stride {
                let a = fs[j];
                let b = a + fs[j + stride];
                let x = ws[j];
                let y = x + ws[j + stride];
                c[0] += a * x;
                c[1] += a * y + b * x;
                c[2] += b * y;
            }
            c
        })
        .reduce(
            || [BF128::ZERO; 3],
            |a, b| std::array::from_fn(|i| a[i] + b[i]),
        );
    SumcheckPolynomial(coeff)
}
pub fn fold_table(table: &mut Vec<BF128>, bit: usize, alpha: BF128) {
    let stride = 1 << bit;
    table.par_chunks_mut(2 * stride).for_each(|v| {
        for j in 0..stride {
            v[j] += alpha * (v[j] + v[j + stride]);
        }
    });
    for group in 0..table.len() / (2 * stride) {
        table.copy_within(
            group * 2 * stride..group * 2 * stride + stride,
            group * stride,
        );
    }
    table.truncate(table.len() / 2);
}
