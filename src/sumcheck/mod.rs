pub mod prover;
use crate::BF128;
use p3_field::PrimeCharacteristicRing;
use p3_maybe_rayon::prelude::*;
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

/// Low-bit-first tensor factor. In characteristic two, x*(1+a) = x + x*a.
fn eq_factor(point: &[BF128], scale: BF128) -> Vec<BF128> {
    let mut out = vec![BF128::ZERO; 1 << point.len()];
    out[0] = scale;
    for (bit, &a) in point.iter().enumerate() {
        let half = 1 << bit;
        let (lo, hi) = out[..2 * half].split_at_mut(half);
        for (x, y) in lo.iter_mut().zip(hi) {
            *y = *x * a;
            *x += *y;
        }
    }
    out
}

/// Build only one full-sized table for a batch of scaled equality claims.
/// The first coordinate always corresponds to the least significant index bit.
pub(crate) fn combined_weights(terms: &[(&[BF128], BF128)]) -> Vec<BF128> {
    assert!(!terms.is_empty());
    let variables = terms[0].0.len();
    assert!(terms.iter().all(|(point, _)| point.len() == variables));
    let split = variables / 2;
    let factors: Vec<_> = terms
        .iter()
        .map(|&(point, scale)| {
            (
                eq_factor(&point[..split], BF128::ONE),
                eq_factor(&point[split..], scale),
            )
        })
        .collect();
    let width = 1 << split;
    let mut out = vec![BF128::ZERO; 1 << variables];
    let fill = |j: usize, row: &mut [BF128]| {
        for (low, high) in &factors {
            for (x, &l) in row.iter_mut().zip(low) {
                *x += l * high[j];
            }
        }
    };
    if out.len() < 4096 {
        for (j, row) in out.chunks_mut(width).enumerate() {
            fill(j, row);
        }
    } else {
        out.par_chunks_mut(width)
            .enumerate()
            .for_each(|(j, row)| fill(j, row));
    }
    out
}

pub fn eq_table(z: &[BF128]) -> Vec<BF128> {
    if z.len() <= 10 {
        eq_factor(z, BF128::ONE)
    } else {
        combined_weights(&[(z, BF128::ONE)])
    }
}

/// Merge dense transparent claims by output block, without a full-table pass per claim.
pub(crate) fn add_scaled_tables(weight: &mut [BF128], terms: &[(&[BF128], BF128)]) {
    assert!(terms.iter().all(|(table, _)| table.len() == weight.len()));
    let add = |start: usize, block: &mut [BF128]| {
        let end = start + block.len();
        for &(table, scale) in terms {
            for (x, &y) in block.iter_mut().zip(&table[start..end]) {
                *x += scale * y;
            }
        }
    };
    if weight.len() < 4096 {
        add(0, weight);
    } else {
        weight
            .par_chunks_mut(1024)
            .enumerate()
            .for_each(|(j, block)| {
                add(j * 1024, block);
            });
    }
}

pub fn evaluate(table: &[BF128], point: &[BF128]) -> BF128 {
    evaluate_base(table, point)
}

/// Evaluate with two sqrt(N)-sized equality factors, without copying/promoting the table.
pub fn evaluate_base<B: p3_field::Field>(table: &[B], point: &[BF128]) -> BF128
where
    BF128: p3_field::ExtensionField<B>,
{
    assert_eq!(table.len(), 1 << point.len());
    let split = point.len() / 2;
    let low = eq_factor(&point[..split], BF128::ONE);
    let high = eq_factor(&point[split..], BF128::ONE);
    let dot = |row: &[B], h: BF128| row.iter().zip(&low).map(|(&f, &w)| w * f).sum::<BF128>() * h;
    if table.len() < 4096 {
        table
            .chunks(low.len())
            .zip(high)
            .map(|(row, h)| dot(row, h))
            .sum()
    } else {
        table
            .par_chunks(low.len())
            .zip(high.par_iter())
            .map(|(row, &h)| dot(row, h))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_binary_field::TowerLevel;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn combined_and_dense_weights_preserve_scales_and_bit_order() {
        let mut rng = ChaCha8Rng::seed_from_u64(757);
        for mu in [0, 1, 3, 8, 13] {
            let points: Vec<Vec<_>> = (0..5)
                .map(|_| (0..mu).map(|_| BF128::from_repr(rng.random())).collect())
                .collect();
            let scales = [
                BF128::ZERO,
                BF128::ONE,
                BF128::from_repr(rng.random()),
                BF128::ONE,
                BF128::ZERO,
            ];
            let terms: Vec<_> = points
                .iter()
                .zip(scales)
                .map(|(p, s)| (p.as_slice(), s))
                .collect();
            let got = combined_weights(&terms);
            for (index, &actual) in got.iter().enumerate() {
                let bits: Vec<_> = (0..mu)
                    .map(|bit| BF128::from_repr(((index >> bit) & 1) as u128))
                    .collect();
                let want: BF128 = terms.iter().map(|&(p, s)| s * eq(p, &bits)).sum();
                assert_eq!(actual, want);
            }
            let tables: Vec<_> = points.iter().map(|p| eq_table(p)).collect();
            let terms: Vec<_> = tables
                .iter()
                .zip(scales)
                .map(|(p, s)| (p.as_slice(), s))
                .collect();
            let initial = BF128::from_repr(rng.random());
            let mut merged = vec![initial; got.len()];
            add_scaled_tables(&mut merged, &terms);
            for (actual, expected) in merged.iter().zip(got) {
                assert_eq!(*actual, initial + expected);
            }
        }
    }
}
