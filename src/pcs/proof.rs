use super::multiproof::{MultiPath, path_bytes};
use crate::{BF128, sumcheck::SumcheckPolynomial};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryOpening {
    pub values: Vec<BF128>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoundProof {
    pub polynomials: Vec<SumcheckPolynomial>,
    pub root: Option<[u8; 32]>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperFoldProof {
    pub rounds: Vec<RoundProof>,
    pub final_table: Vec<BF128>,
    pub queries: Vec<Vec<QueryOpening>>,
    /// One deduplicated authentication path per oracle, top to bottom.
    pub paths: Vec<MultiPath>,
}
impl HyperFoldProof {
    /// Exact field/hash payload, excluding container lengths. Commitment counted separately.
    pub fn size_in_bytes(&self) -> usize {
        self.rounds
            .iter()
            .map(|r| 48 * r.polynomials.len() + 32 * usize::from(r.root.is_some()))
            .sum::<usize>()
            + 16 * self.final_table.len()
            + self
                .queries
                .iter()
                .flatten()
                .map(|q| 16 * q.values.len())
                .sum::<usize>()
            + path_bytes(&self.paths)
    }
    /// Actual chosen serde wire framing, provided separately from paper payload.
    pub fn wire_size_in_bytes(&self) -> usize {
        bincode::serialized_size(self).unwrap() as usize
    }
}
