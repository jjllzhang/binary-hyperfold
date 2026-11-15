use crate::BF128;
use p3_binary_dft::{AdditiveNtt, LchNtt};
use p3_matrix::dense::RowMajorMatrix;
/// Message entries are novel-basis polynomial coefficients (a linear MDS encoder).
pub fn encode(message: &[BF128], log_rate: usize) -> Vec<BF128> {
    LchNtt::default()
        .ntt_batch_borrowed(RowMajorMatrix::new(message.to_vec(), 1).as_view(), log_rate)
        .values
}
