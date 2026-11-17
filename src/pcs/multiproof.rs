//! Transcript-selected query indices and shared Merkle authentication paths.
//!
//! As in Whir's Merkle multi-openings
//! (https://github.com/worldfnd/whir/blob/main/src/protocols/merkle_tree.rs),
//! each layer sends only boundary siblings
//! of the queried-leaf frontier. Plonky3's `open_multi_batch`/
//! `verify_multi_batch` implement this pruning (and check duplicate rows).
use p3_merkle_tree::PrunedMerklePaths;

use crate::transcript::ProverTranscript;

pub type MultiPath = PrunedMerklePaths<u8, 32>;

/// One list of row indices per oracle, ordered from top to bottom.
/// The transcript samples top indices only; lower indices are their remainders.
pub(crate) fn layer_indices(
    tr: &mut ProverTranscript,
    heights: &[usize],
    count: usize,
) -> Vec<Vec<usize>> {
    let mut indices = vec![Vec::with_capacity(count); heights.len()];
    for _ in 0..count {
        let mut j = tr.index(*heights.last().unwrap());
        for layer in (0..heights.len()).rev() {
            indices[heights.len() - 1 - layer].push(j);
            if layer > 0 {
                j %= heights[layer - 1];
            }
        }
    }
    indices
}

pub(crate) fn path_bytes(paths: &[MultiPath]) -> usize {
    32 * paths.iter().map(|p| p.sibling_hashes.len()).sum::<usize>()
}
