use super::SumcheckPolynomial;
use crate::BF128;
use p3_field::PrimeCharacteristicRing;
use p3_maybe_rayon::prelude::*;
pub fn polynomial(f: &[BF128], w: &[BF128], bit: usize) -> SumcheckPolynomial {
    let stride = 1 << bit;
    let coeff = (0..f.len() / 2)
        .into_par_iter()
        .fold(
            || [BF128::ZERO; 3],
            |mut c, pair| {
                let j = (pair / stride) * (2 * stride) + pair % stride;
                let a = f[j];
                let b = a + f[j + stride];
                let x = w[j];
                let y = x + w[j + stride];
                c[0] += a * x;
                c[1] += a * y + b * x;
                c[2] += b * y;
                c
            },
        )
        .reduce(
            || [BF128::ZERO; 3],
            |a, b| std::array::from_fn(|i| a[i] + b[i]),
        );
    SumcheckPolynomial(coeff)
}
pub fn fold_table(table: &mut Vec<BF128>, bit: usize, alpha: BF128) {
    let stride = 1 << bit;
    table.par_chunks_mut(2 * stride).for_each(|v| {
        let (lo, hi) = v.split_at_mut(stride);
        if stride >= 1024 {
            lo.par_iter_mut()
                .zip(hi.par_iter())
                .for_each(|(x, &y)| *x += alpha * (*x + y));
        } else {
            for (x, &y) in lo.iter_mut().zip(hi.iter()) {
                *x += alpha * (*x + y);
            }
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

pub fn polynomial_base<B: p3_field::Field>(f: &[B], w: &[BF128], bit: usize) -> SumcheckPolynomial
where
    BF128: p3_field::ExtensionField<B>,
{
    let stride = 1 << bit;
    let coeff = (0..f.len() / 2)
        .into_par_iter()
        .fold(
            || [BF128::ZERO; 3],
            |mut c, pair| {
                let j = (pair / stride) * (2 * stride) + pair % stride;
                let a = f[j];
                let b = a + f[j + stride];
                let x = w[j];
                let y = x + w[j + stride];
                c[0] += x * a;
                c[1] += y * a + x * b;
                c[2] += y * b;
                c
            },
        )
        .reduce(
            || [BF128::ZERO; 3],
            |a, b| std::array::from_fn(|i| a[i] + b[i]),
        );
    SumcheckPolynomial(coeff)
}

/// Bind a variable from a native base table directly into a half-size extension table.
pub fn fold_table_base<B: p3_field::Field>(table: &[B], bit: usize, alpha: BF128) -> Vec<BF128>
where
    BF128: p3_field::ExtensionField<B>,
{
    let stride = 1 << bit;
    (0..table.len() / 2)
        .into_par_iter()
        .map(|pair| {
            let j = (pair / stride) * (2 * stride) + pair % stride;
            BF128::from(table[j]) + alpha * (table[j] + table[j + stride])
        })
        .collect()
}
