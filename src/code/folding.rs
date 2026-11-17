use super::{HyperFoldableCode, base_code};
use crate::BF128;
use p3_commit::RowSink;
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
    fold_into_layout_with_sink(len, width, fold_symbol, None)
}

pub(crate) fn fold_into_layout_with_sink(
    len: usize,
    width: usize,
    fold_symbol: impl Fn(usize) -> BF128 + Send + Sync,
    sink: Option<&RowSink<'_, BF128>>,
) -> RowMajorMatrix<BF128> {
    let height = len / width;
    finish_rows(
        vec![BF128::ZERO; len],
        width,
        |j, row| {
            for (a, value) in row.iter_mut().enumerate() {
                *value = fold_symbol(j + a * height);
            }
        },
        sink,
    )
}

/// Finish and immediately hand over cache-sized blocks, not individual rows.
/// The output allocation remains the oracle retained by the MMCS for openings.
/// In particular, the sink is never called inside a row's arithmetic kernel.
pub(crate) fn finish_rows<T: Clone + Send + Sync>(
    mut values: Vec<T>,
    width: usize,
    finish_row: impl Fn(usize, &mut [T]) + Send + Sync,
    sink: Option<&RowSink<'_, T>>,
) -> RowMajorMatrix<T> {
    assert!(width > 0);
    assert_eq!(values.len() % width, 0);
    if sink.is_none() {
        // Preserve the recursive encoder's existing fine-grained scheduling.
        values
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(j, row)| finish_row(j, row));
        return RowMajorMatrix::new(values, width);
    }
    let cache_rows = (256 * 1024 / (size_of::<T>().max(1) * width)).max(1);
    let height = values.len() / width;
    // Small intermediate oracles must still expose enough blocks to the pool.
    // A fixed cache-sized tile would serialize their arithmetic and leaf hashing.
    let task_rows = height
        .div_ceil(current_num_threads() * 4)
        .max(32)
        .next_power_of_two();
    let block_rows = cache_rows.min(task_rows);
    values
        .par_chunks_mut(block_rows * width)
        .enumerate()
        .for_each(|(block, rows)| {
            let first_row = block * block_rows;
            for (offset, row) in rows.chunks_exact_mut(width).enumerate() {
                finish_row(first_row + offset, row);
            }
            if let Some(sink) = sink {
                sink(first_row, rows);
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
        self.encode_impl(layer, message, None)
    }

    /// Encode as usual, handing every final output row to `sink` exactly once.
    /// Recursive child oracles are not handed over; only the requested layer is.
    pub fn encode_with_sink(
        &self,
        layer: usize,
        message: &[BF128],
        sink: &RowSink<'_, BF128>,
    ) -> RowMajorMatrix<BF128> {
        self.encode_impl(layer, message, Some(sink))
    }

    fn encode_impl(
        &self,
        layer: usize,
        message: &[BF128],
        sink: Option<&RowSink<'_, BF128>>,
    ) -> RowMajorMatrix<BF128> {
        assert_eq!(message.len(), self.k[layer]);
        if layer == 0 {
            return if let Some(sink) = sink {
                base_code::encode_with_sink(message, self.log_rate, sink)
            } else {
                RowMajorMatrix::new(base_code::encode(message, self.log_rate), 1)
            };
        }
        let children: Vec<_> = message
            .par_chunks(self.k[layer - 1])
            .map(|m| self.encode(layer - 1, m))
            .collect();
        finish_rows(
            vec![BF128::ZERO; self.n[layer]],
            self.r,
            |j, row| {
                row[0] = symbol(&children[0], j);
                for a in 1..self.r {
                    row[a] = symbol(&children[a], j) * self.twists[layer - 1][j * self.r + a];
                }
                self.matrix.apply(row, false);
            },
            sink,
        )
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
    /// Fold into the destination layout and hand over each finished block once.
    pub fn fold_with_sink(
        &self,
        layer: usize,
        word: &RowMajorMatrix<BF128>,
        beta: &[BF128],
        sink: &RowSink<'_, BF128>,
    ) -> RowMajorMatrix<BF128> {
        fold_into_layout_with_sink(
            word.values.len() / self.r,
            if layer == 1 { 1 } else { self.r },
            |j| self.fold_row(layer, j, &word.values[j * self.r..(j + 1) * self.r], beta),
            Some(sink),
        )
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
