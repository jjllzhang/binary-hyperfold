pub mod error;
mod multiproof;
pub mod proof;
pub mod prover;
pub mod verifier;
pub mod weighted;
use crate::{
    BF128,
    code::{
        HyperFoldableCode,
        native::{NativeCode, NativeField},
    },
    parameters::HyperFoldConfig,
    transcript::ProverTranscript,
};
use p3_blake3::Blake3;
use p3_commit::Mmcs;
use p3_field::ExtensionField;
use p3_matrix::{Dimensions, dense::RowMajorMatrix};
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
/// Encode and commit the top oracle, retaining its matrix for openings.
/// Stream large outputs; on small outputs the sink's per-row atomic bookkeeping
/// can outweigh cache reuse (notably r=8), so keep the existing batch builder.
pub fn commit_encoded(
    code: &HyperFoldableCode,
    layer: usize,
    message: &[BF128],
) -> (<Merkle as Mmcs<BF128>>::Commitment, Tree) {
    if code.n[layer] * size_of::<BF128>() < 64 * 1024 * 1024 {
        return mmcs().commit_matrix(code.encode(layer, message));
    }
    mmcs().commit_matrix_streamed(oracle_dimensions(code.n[layer], code.r, layer), |sink| {
        code.encode_with_sink(layer, message, sink)
    })
}

/// Fold layer `layer` into layer `layer - 1` and hash each finished block there.
/// `beta` contains equality weights, just as for `HyperFoldableCode::fold`.
pub fn commit_folded(
    code: &HyperFoldableCode,
    layer: usize,
    word: &RowMajorMatrix<BF128>,
    beta: &[BF128],
) -> (<Merkle as Mmcs<BF128>>::Commitment, Tree) {
    assert!(layer > 0);
    mmcs().commit_matrix_streamed(
        oracle_dimensions(code.n[layer - 1], code.r, layer - 1),
        |sink| code.fold_with_sink(layer, word, beta, sink),
    )
}

/// Opt-in native top streaming with the caller's MMCS and unchanged leaf bytes.
/// `NativePcs::commit` keeps the batch path, which was faster in measured sizes.
pub fn commit_native_encoded<B, MT>(
    mmcs: &MT,
    code: &NativeCode<B>,
    layer: usize,
    message: &[B],
) -> (MT::Commitment, MT::ProverData<RowMajorMatrix<B>>)
where
    B: NativeField,
    BF128: ExtensionField<B>,
    MT: Mmcs<B>,
{
    mmcs.commit_matrix_streamed(oracle_dimensions(code.n[layer], code.r, layer), |sink| {
        code.encode_with_sink(layer, message, sink)
    })
}

/// Native top folding keeps H inversion and untwisting in the base field;
/// only the completed destination oracle and its leaf hashes use BF128.
pub fn commit_native_folded_top<B: NativeField>(
    code: &NativeCode<B>,
    layer: usize,
    word: &RowMajorMatrix<B>,
    beta: &[BF128],
) -> (<Merkle as Mmcs<BF128>>::Commitment, Tree)
where
    BF128: ExtensionField<B>,
{
    assert!(layer > 0);
    mmcs().commit_matrix_streamed(
        oracle_dimensions(code.n[layer - 1], code.r, layer - 1),
        |sink| code.fold_top_with_sink(layer, word, beta, sink),
    )
}

/// Stream an intermediate BF128 fold of a native-field code into the usual MMCS.
pub fn commit_native_folded<B: NativeField>(
    code: &NativeCode<B>,
    layer: usize,
    word: &RowMajorMatrix<BF128>,
    beta: &[BF128],
) -> (<Merkle as Mmcs<BF128>>::Commitment, Tree)
where
    BF128: ExtensionField<B>,
{
    assert!(layer > 0);
    mmcs().commit_matrix_streamed(
        oracle_dimensions(code.n[layer - 1], code.r, layer - 1),
        |sink| code.fold_with_sink(layer, word, beta, sink),
    )
}

fn oracle_dimensions(len: usize, arity: usize, layer: usize) -> Dimensions {
    let width = if layer == 0 { 1 } else { arity };
    Dimensions {
        width,
        height: len / width,
    }
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

pub mod native;
pub use native as native64;
pub use native::{BinaryHyperFoldPcs16, BinaryHyperFoldPcs32, BinaryHyperFoldPcs64};
