pub mod error;
pub mod proof;
pub mod prover;
pub mod verifier;
use crate::{
    BF128, code::HyperFoldableCode, parameters::HyperFoldConfig, transcript::ProverTranscript,
};
use p3_blake3::Blake3;
use p3_commit::Mmcs;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, MerkleCap, SerializingHasher};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
pub type Merkle = MerkleTreeMmcs<
    BF128,
    u8,
    SerializingHasher<Blake3>,
    CompressionFunctionFromHasher<Blake3, 2, 32>,
    2,
    32,
>;
pub type Tree = <Merkle as Mmcs<BF128>>::ProverData<RowMajorMatrix<BF128>>;
pub fn mmcs() -> Merkle {
    Merkle::new(
        SerializingHasher::new(Blake3),
        CompressionFunctionFromHasher::new(Blake3),
        0,
    )
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Commitment {
    pub root: [u8; 32],
    pub z0: Vec<BF128>,
    pub y0: BF128,
}
impl Commitment {
    pub fn size_in_bytes(&self) -> usize {
        32 + 16 * (self.z0.len() + 1)
    }
}
pub struct ProverData {
    pub message: Vec<BF128>,
    pub tree: Tree,
    pub commitment: Commitment,
}
#[derive(Clone, Debug)]
pub struct BinaryHyperFoldPcs {
    pub config: HyperFoldConfig,
    pub code: HyperFoldableCode,
    pub seed: [u8; 32],
}
impl BinaryHyperFoldPcs {
    pub fn setup(config: HyperFoldConfig, seed: [u8; 32]) -> Self {
        assert!(config.d > 0);
        let code = HyperFoldableCode::setup(&config, &mut ChaCha8Rng::from_seed(seed));
        Self { config, code, seed }
    }
    pub(crate) fn transcript(&self) -> ProverTranscript {
        let mut tr = ProverTranscript::new(b"binary-hyperfold-v1");
        tr.observe_bytes(&bincode::serialize(&self.config).unwrap());
        tr.observe_bytes(&self.seed);
        tr
    }
    pub(crate) fn commitment_transcript(&self, cm: &Commitment) -> Option<ProverTranscript> {
        let mut tr = self.transcript();
        tr.observe_bytes(&cm.root);
        if tr.point(self.config.params.mu) != cm.z0 {
            return None;
        }
        tr.observe(cm.y0);
        Some(tr)
    }
    pub(crate) fn claim_transcript(
        &self,
        cm: &Commitment,
        points: &[Vec<BF128>],
        values: &[BF128],
    ) -> Option<ProverTranscript> {
        let mut tr = self.commitment_transcript(cm)?;
        tr.observe_bytes(&(points.len() as u64).to_le_bytes());
        for (p, &y) in points.iter().zip(values) {
            tr.observe_fields(p);
            tr.observe(y)
        }
        Some(tr)
    }
    pub(crate) fn query_count(&self, m: usize) -> usize {
        let c = &self.config;
        let extra = (m.saturating_sub(1) * c.list_size) as f64 / 2f64.powi(128);
        crate::parameters::soundness::query_count(
            c.params.security_bits,
            c.epsilon_code,
            c.epsilon_field + extra,
            c.delta,
        )
        .expect("no batch soundness budget")
    }
}
pub(crate) fn cap(root: [u8; 32]) -> MerkleCap<BF128, [u8; 32]> {
    MerkleCap::new(vec![root])
}
