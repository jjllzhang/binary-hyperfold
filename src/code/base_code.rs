use crate::BF128;
use p3_binary_dft::{AdditiveNtt, LchNtt};
use p3_matrix::dense::RowMajorMatrixView;
/// Message entries are novel-basis polynomial coefficients (a linear MDS encoder).
pub fn encode(message: &[BF128], log_rate: usize) -> Vec<BF128> {
    LchNtt::default()
        .ntt_batch_borrowed(RowMajorMatrixView::new(message, 1), log_rate)
        .values
}

/// Same Cantor evaluation domain, computed natively over the base field.
pub fn encode_64(message: &[crate::BF64], log_rate: usize) -> Vec<crate::BF64> {
    LchNtt::default()
        .ntt_batch_borrowed(RowMajorMatrixView::new(message, 1), log_rate)
        .values
}

/// Shared Cantor-domain encoder, monomorphized over each native tower level.
pub fn encode_native<B: p3_binary_dft::ButterflyField>(message: &[B], log_rate: usize) -> Vec<B> {
    LchNtt::default()
        .ntt_batch_borrowed(RowMajorMatrixView::new(message, 1), log_rate)
        .values
}
