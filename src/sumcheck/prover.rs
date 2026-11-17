use super::SumcheckPolynomial;
use crate::BF128;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_maybe_rayon::prelude::*;

fn polynomial_impl<B: Field, const KNOWN_SUM: bool>(
    f: &[B],
    w: &[BF128],
    bit: usize,
    sum: BF128,
) -> SumcheckPolynomial
where
    BF128: ExtensionField<B>,
{
    let stride = 1usize << bit;
    assert_eq!(f.len(), w.len());
    assert!(f.len().is_multiple_of(2 * stride));
    let mut coeff = (0..f.len() / 2).into_par_iter().par_fold_reduce(
        || [BF128::ZERO; 3],
        |mut c, pair| {
            let j = ((pair >> bit) << (bit + 1)) + (pair & (stride - 1));
            let a = f[j];
            let b = a + f[j + stride];
            let x = w[j];
            let y = x + w[j + stride];
            c[0] += x * a;
            if !KNOWN_SUM {
                c[1] += y * a + x * b;
            }
            c[2] += y * b;
            c
        },
        |a, b| std::array::from_fn(|i| a[i] + b[i]),
    );
    if KNOWN_SUM {
        // p(0)+p(1) = c1+c2 in characteristic two. Keep the original
        // three-coefficient message and transcript, but measure only c0/c2.
        coeff[1] = sum + coeff[2];
    }
    SumcheckPolynomial(coeff)
}

pub fn polynomial(f: &[BF128], w: &[BF128], bit: usize) -> SumcheckPolynomial {
    polynomial_impl::<_, false>(f, w, bit, BF128::ZERO)
}

/// The caller supplies the current inner-product claim of f and w.
pub fn polynomial_with_sum(f: &[BF128], w: &[BF128], bit: usize, sum: BF128) -> SumcheckPolynomial {
    polynomial_impl::<_, true>(f, w, bit, sum)
}

pub fn polynomial_base<B: Field>(f: &[B], w: &[BF128], bit: usize) -> SumcheckPolynomial
where
    BF128: ExtensionField<B>,
{
    polynomial_impl::<_, false>(f, w, bit, BF128::ZERO)
}

/// Native first-round counterpart of `polynomial_with_sum`.
pub fn polynomial_base_with_sum<B: Field>(
    f: &[B],
    w: &[BF128],
    bit: usize,
    sum: BF128,
) -> SumcheckPolynomial
where
    BF128: ExtensionField<B>,
{
    polynomial_impl::<_, true>(f, w, bit, sum)
}

/// Write the compact result into reusable disjoint storage. Parallel in-place
/// compaction would race with other groups' source reads.
pub fn fold_table_with_scratch(
    table: &mut Vec<BF128>,
    bit: usize,
    alpha: BF128,
    scratch: &mut Vec<BF128>,
) {
    let stride = 1usize << bit;
    assert!(table.len().is_multiple_of(2 * stride));
    let half = table.len() / 2;
    scratch.resize(half, BF128::ZERO);
    let fill = |start: usize, out: &mut [BF128]| {
        for (offset, x) in out.iter_mut().enumerate() {
            let pair = start + offset;
            let j = ((pair >> bit) << (bit + 1)) + (pair & (stride - 1));
            *x = table[j] + alpha * (table[j] + table[j + stride]);
        }
    };
    if half < 2048 {
        fill(0, scratch);
    } else {
        scratch
            .par_chunks_mut(1024)
            .enumerate()
            .for_each(|(j, out)| {
                fill(j * 1024, out);
            });
    }
    std::mem::swap(table, scratch);
}

pub fn fold_table(table: &mut Vec<BF128>, bit: usize, alpha: BF128) {
    fold_table_with_scratch(table, bit, alpha, &mut Vec::new());
}

/// Bind a native base table directly into a half-size extension table.
pub fn fold_table_base<B: Field>(table: &[B], bit: usize, alpha: BF128) -> Vec<BF128>
where
    BF128: ExtensionField<B>,
{
    let stride = 1usize << bit;
    assert!(table.len().is_multiple_of(2 * stride));
    let mut out = vec![BF128::ZERO; table.len() / 2];
    let fill = |start: usize, out: &mut [BF128]| {
        for (offset, x) in out.iter_mut().enumerate() {
            let pair = start + offset;
            let j = ((pair >> bit) << (bit + 1)) + (pair & (stride - 1));
            *x = BF128::from(table[j]) + alpha * (table[j] + table[j + stride]);
        }
    };
    if out.len() < 2048 {
        fill(0, &mut out);
    } else {
        out.par_chunks_mut(1024).enumerate().for_each(|(j, block)| {
            fill(j * 1024, block);
        });
    }
    out
}
