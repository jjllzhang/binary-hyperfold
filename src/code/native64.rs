//! Native base-field storage; extension folding starts only after challenges.
use super::{base_code, folding::from_symbols, matrix_h::*};
use crate::{
    BF64, BF128,
    parameters::{HyperFoldConfig, MatrixKind},
};
use p3_binary_field::TowerLevel;
use p3_field::{PrimeCharacteristicRing, batch_multiplicative_inverse};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use rand::{Rng, RngExt};

#[derive(Clone, Debug)]
pub struct Code64 {
    pub r: usize,
    pub k: Vec<usize>,
    pub n: Vec<usize>,
    pub log_rate: usize,
    pub twists: Vec<Vec<BF64>>,
    pub inverse_twists: Vec<Vec<BF64>>,
}
pub fn apply_h64(x: &mut [BF64], inv: bool) {
    match (x.len(), inv) {
        (2, false) => apply_h2_64(x),
        (2, true) => apply_h2_inv_64(x),
        (4, false) => apply_h4_64(x),
        (4, true) => apply_h4_inv_64(x),
        (8, _) => apply_h8_64(x),
        _ => panic!("arity"),
    }
}
pub fn apply_h128(x: &mut [BF128], inv: bool) {
    match (x.len(), inv) {
        (2, false) => apply_h2(x),
        (2, true) => apply_h2_inv(x),
        (4, false) => apply_h4(x),
        (4, true) => apply_h4_inv(x),
        (8, _) => apply_h8(x),
        _ => panic!("arity"),
    }
}
fn symbol(word: &RowMajorMatrix<BF64>, j: usize) -> BF64 {
    if word.width == 1 {
        word.values[j]
    } else {
        let n = word.values.len() / word.width;
        word.values[(j % n) * word.width + j / n]
    }
}
impl Code64 {
    pub fn setup(c: &HyperFoldConfig, rng: &mut impl Rng) -> Self {
        assert_eq!(c.params.base_field_bits, 64);
        assert_eq!(c.params.matrix_kind, MatrixKind::Specialized);
        let mut twists = vec![];
        let mut inverse_twists = vec![];
        for i in 1..=c.d {
            let mut v = vec![BF64::ONE; c.n[i]];
            for row in v.chunks_mut(c.params.r) {
                for x in &mut row[1..] {
                    loop {
                        // Deliberately consume the exact u128 RNG stream of the baseline.
                        let bits = rng.random::<u128>() as u64;
                        if bits != 0 {
                            *x = BF64::from_repr(bits);
                            break;
                        }
                    }
                }
            }
            inverse_twists.push(batch_multiplicative_inverse(&v));
            twists.push(v);
        }
        Self {
            r: c.params.r,
            k: c.k.clone(),
            n: c.n.clone(),
            log_rate: c.params.c.ilog2() as usize,
            twists,
            inverse_twists,
        }
    }
    pub fn encode(&self, layer: usize, message: &[BF64]) -> RowMajorMatrix<BF64> {
        assert_eq!(message.len(), self.k[layer]);
        if layer == 0 {
            return RowMajorMatrix::new(base_code::encode_64(message, self.log_rate), 1);
        }
        let children: Vec<_> = message
            .par_chunks(self.k[layer - 1])
            .map(|m| self.encode(layer - 1, m))
            .collect();
        let mut values = vec![BF64::ZERO; self.n[layer]];
        values
            .par_chunks_mut(self.r)
            .enumerate()
            .for_each(|(j, row)| {
                row[0] = symbol(&children[0], j);
                for a in 1..self.r {
                    row[a] = symbol(&children[a], j) * self.twists[layer - 1][j * self.r + a];
                }
                apply_h64(row, false);
            });
        RowMajorMatrix::new(values, self.r)
    }
    pub fn fold_row_top(&self, layer: usize, j: usize, row: &[BF64], beta: &[BF128]) -> BF128 {
        let mut v = [BF64::ZERO; 8];
        v[..self.r].copy_from_slice(row);
        apply_h64(&mut v[..self.r], true);
        (0..self.r)
            .map(|a| beta[a] * (self.inverse_twists[layer - 1][j * self.r + a] * v[a]))
            .sum()
    }
    pub fn fold_row(&self, layer: usize, j: usize, row: &[BF128], beta: &[BF128]) -> BF128 {
        let mut v = [BF128::ZERO; 8];
        v[..self.r].copy_from_slice(row);
        apply_h128(&mut v[..self.r], true);
        (0..self.r)
            .map(|a| beta[a] * (v[a] * self.inverse_twists[layer - 1][j * self.r + a]))
            .sum()
    }
    pub fn fold_top(
        &self,
        layer: usize,
        word: &RowMajorMatrix<BF64>,
        beta: &[BF128],
    ) -> RowMajorMatrix<BF128> {
        let values = word
            .values
            .par_chunks(self.r)
            .enumerate()
            .map(|(j, row)| self.fold_row_top(layer, j, row, beta))
            .collect();
        from_symbols(values, if layer == 1 { 1 } else { self.r })
    }
    pub fn fold(
        &self,
        layer: usize,
        word: &RowMajorMatrix<BF128>,
        beta: &[BF128],
    ) -> RowMajorMatrix<BF128> {
        let values = word
            .values
            .par_chunks(self.r)
            .enumerate()
            .map(|(j, row)| self.fold_row(layer, j, row, beta))
            .collect();
        from_symbols(values, if layer == 1 { 1 } else { self.r })
    }
}
