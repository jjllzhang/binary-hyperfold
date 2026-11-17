use super::{HyperFoldableCode, base_code};
use crate::BF128;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

/// Low-bit-first equality weights for the r-way split.
pub fn eq_weights(alpha: &[BF128]) -> Vec<BF128> {
    let mut v = vec![BF128::ONE];
    for &a in alpha {
        let old = v.len();
        v.resize(old * 2, BF128::ZERO);
        for j in 0..old {
            let x = v[j];
            v[j] = x * (BF128::ONE + a);
            v[j + old] = x * a;
        }
    }
    v
}
pub fn symbol(word: &RowMajorMatrix<BF128>, index: usize) -> BF128 {
    if word.width == 1 {
        word.values[index]
    } else {
        let n = word.values.len() / word.width;
        word.values[(index % n) * word.width + index / n]
    }
}
pub fn from_symbols(symbols: Vec<BF128>, width: usize) -> RowMajorMatrix<BF128> {
    if width == 1 {
        return RowMajorMatrix::new(symbols, 1);
    }
    let n = symbols.len() / width;
    let mut out = vec![BF128::ZERO; symbols.len()];
    out.par_chunks_mut(width).enumerate().for_each(|(j, row)| {
        for a in 0..width {
            row[a] = symbols[j + a * n]
        }
    });
    RowMajorMatrix::new(out, width)
}
/// Fold directly into the destination's row-major layout. Logical symbol
/// `j + a * height` belongs in lane `a` of destination row `j`.
pub(crate) fn fold_into_layout(
    len: usize,
    width: usize,
    fold_symbol: impl Fn(usize) -> BF128 + Send + Sync,
) -> RowMajorMatrix<BF128> {
    let height = len / width;
    let mut values = vec![BF128::ZERO; len];
    values
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(j, row)| {
            for (a, value) in row.iter_mut().enumerate() {
                *value = fold_symbol(j + a * height);
            }
        });
    RowMajorMatrix::new(values, width)
}

pub(crate) fn dot_product(r: usize, a: &[BF128], b: &[BF128]) -> BF128 {
    match r {
        2 => BF128::dot_product::<2>(a[..2].try_into().unwrap(), b[..2].try_into().unwrap()),
        4 => BF128::dot_product::<4>(a[..4].try_into().unwrap(), b[..4].try_into().unwrap()),
        8 => BF128::dot_product::<8>(a[..8].try_into().unwrap(), b[..8].try_into().unwrap()),
        _ => panic!("arity"),
    }
}

impl HyperFoldableCode {
    pub fn encode(&self, layer: usize, message: &[BF128]) -> RowMajorMatrix<BF128> {
        assert_eq!(message.len(), self.k[layer]);
        if layer == 0 {
            return RowMajorMatrix::new(base_code::encode(message, self.log_rate), 1);
        }
        let children: Vec<_> = message
            .par_chunks(self.k[layer - 1])
            .map(|m| self.encode(layer - 1, m))
            .collect();
        let mut values = vec![BF128::ZERO; self.n[layer]];
        values
            .par_chunks_mut(self.r)
            .enumerate()
            .for_each(|(j, row)| {
                row[0] = symbol(&children[0], j);
                for a in 1..self.r {
                    row[a] = symbol(&children[a], j) * self.twists[layer - 1][j * self.r + a];
                }
                self.matrix.apply(row, false);
            });
        RowMajorMatrix::new(values, self.r)
    }
    pub fn fold_row(&self, layer: usize, j: usize, row: &[BF128], beta: &[BF128]) -> BF128 {
        let mut v = [BF128::ZERO; 8];
        v[..self.r].copy_from_slice(row);
        self.matrix.apply(&mut v[..self.r], true);
        // Lane zero is untwisted by construction.
        for (a, value) in v.iter_mut().enumerate().take(self.r).skip(1) {
            *value *= self.inverse_twists[layer - 1][j * self.r + a];
        }
        dot_product(self.r, beta, &v)
    }
    pub fn fold(
        &self,
        layer: usize,
        word: &RowMajorMatrix<BF128>,
        beta: &[BF128],
    ) -> RowMajorMatrix<BF128> {
        fold_into_layout(
            word.values.len() / self.r,
            if layer == 1 { 1 } else { self.r },
            |j| self.fold_row(layer, j, &word.values[j * self.r..(j + 1) * self.r], beta),
        )
    }
}
