//! Single-claim PCS with a native BF16/BF32/BF64 message and top oracle.
//! Intermediate oracles and all challenges remain BF128. Top leaves hash native-width symbols.
use super::{
    Commitment, Tree, cap,
    error::VerifierError,
    mmcs,
    proof::{QueryOpening, RoundProof},
};
use crate::{
    BF16, BF32, BF64, BF128,
    code::{
        base_code,
        folding::eq_weights,
        native::{NativeCode, NativeField},
    },
    parameters::HyperFoldConfig,
    sumcheck::{
        self,
        prover::{fold_table, fold_table_base, polynomial, polynomial_base},
    },
    transcript::ProverTranscript,
};
use p3_blake3::Blake3;
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::ExtensionField;
use p3_matrix::{Dimensions, dense::RowMajorMatrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, MerkleCap, SerializingHasher};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

type MerkleBase<B> = MerkleTreeMmcs<
    B,
    u8,
    SerializingHasher<Blake3>,
    CompressionFunctionFromHasher<Blake3, 2, 32>,
    2,
    32,
>;
pub type TreeBase<B> = <MerkleBase<B> as Mmcs<B>>::ProverData<RowMajorMatrix<B>>;
fn mmcs_base<B: NativeField>() -> MerkleBase<B> {
    MerkleBase::new(
        SerializingHasher::new(Blake3),
        CompressionFunctionFromHasher::new(Blake3),
        0,
    )
}
fn cap_base<B: NativeField>(root: [u8; 32]) -> MerkleCap<B, [u8; 32]> {
    MerkleCap::new(vec![root])
}

pub struct ProverDataNative<B: NativeField> {
    pub message: Vec<B>,
    pub tree: TreeBase<B>,
    pub commitment: Commitment,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct TopOpening<B: NativeField> {
    pub values: Vec<B>,
    pub path: Vec<[u8; 32]>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct QueryNative<B: NativeField> {
    pub top: TopOpening<B>,
    pub rest: Vec<QueryOpening>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NativeProof<B: NativeField> {
    pub rounds: Vec<RoundProof>,
    pub final_table: Vec<BF128>,
    pub queries: Vec<QueryNative<B>>,
}
impl<B: NativeField> NativeProof<B> {
    pub fn size_in_bytes(&self) -> usize {
        self.rounds
            .iter()
            .map(|r| 48 * r.polynomials.len() + 32 * usize::from(r.root.is_some()))
            .sum::<usize>()
            + 16 * self.final_table.len()
            + self
                .queries
                .iter()
                .map(|q| {
                    (B::BITS / 8) * q.top.values.len()
                        + 32 * q.top.path.len()
                        + q.rest
                            .iter()
                            .map(|v| 16 * v.values.len() + 32 * v.path.len())
                            .sum::<usize>()
                })
                .sum::<usize>()
    }
    pub fn wire_size_in_bytes(&self) -> usize {
        bincode::serialized_size(self).unwrap() as usize
    }
}
#[derive(Clone, Debug)]
pub struct NativePcs<B: NativeField> {
    pub config: HyperFoldConfig,
    pub code: NativeCode<B>,
    pub seed: [u8; 32],
}
impl<B: NativeField> NativePcs<B>
where
    BF128: ExtensionField<B>,
{
    pub fn setup(config: HyperFoldConfig, seed: [u8; 32]) -> Self {
        assert!(config.d > 0);
        let code = NativeCode::setup(&config, &mut ChaCha8Rng::from_seed(seed));
        Self { config, code, seed }
    }
    fn transcript(&self) -> ProverTranscript {
        // Distinct leaf encoding is explicitly domain-separated from the baseline.
        let mut tr = ProverTranscript::new(match B::BITS {
            16 => b"binary-hyperfold-bf16-v1",
            32 => b"binary-hyperfold-bf32-v1",
            64 => b"binary-hyperfold-bf64-v1",
            _ => unreachable!(),
        });
        tr.observe_bytes(&bincode::serialize(&self.config).unwrap());
        tr.observe_bytes(&self.seed);
        tr
    }
    fn claim_transcript(
        &self,
        cm: &Commitment,
        point: &[BF128],
        value: BF128,
    ) -> Option<ProverTranscript> {
        let mut tr = self.transcript();
        tr.observe_bytes(&cm.root);
        if tr.point(self.config.params.mu) != cm.z0 {
            return None;
        }
        tr.observe(cm.y0);
        tr.observe_bytes(&1u64.to_le_bytes());
        tr.observe_fields(point);
        tr.observe(value);
        Some(tr)
    }
    pub fn commit(&self, message: Vec<B>) -> (Commitment, ProverDataNative<B>) {
        let (root, tree) =
            mmcs_base::<B>().commit_matrix(self.code.encode(self.config.d, &message));
        let root = root.roots()[0];
        let mut tr = self.transcript();
        tr.observe_bytes(&root);
        let z0 = tr.point(self.config.params.mu);
        let y0 = sumcheck::evaluate_base(&message, &z0);
        let cm = Commitment { root, z0, y0 };
        let data = ProverDataNative {
            message,
            tree,
            commitment: cm.clone(),
        };
        (cm, data)
    }
    pub fn open(
        &self,
        data: &ProverDataNative<B>,
        point: Vec<BF128>,
        value: BF128,
    ) -> NativeProof<B> {
        let c = &self.config;
        assert_eq!(point.len(), c.params.mu);
        debug_assert_eq!(sumcheck::evaluate_base(&data.message, &point), value);
        let mut tr = self
            .claim_transcript(&data.commitment, &point, value)
            .unwrap();
        let gamma = tr.sample();
        let mut weight = sumcheck::eq_table(&data.commitment.z0);
        for (w, e) in weight.iter_mut().zip(sumcheck::eq_table(&point)) {
            *w += gamma * e;
        }
        let mut f = vec![];
        let mut rounds = vec![];
        let mut trees: Vec<Tree> = vec![];
        for i in (1..=c.d).rev() {
            let bit = c.mu_0 + (i - 1) * c.nu;
            let mut polynomials = vec![];
            let mut alpha = vec![];
            for step in 0..c.nu {
                let first = i == c.d && step == 0;
                let p = if first {
                    polynomial_base(&data.message, &weight, bit)
                } else {
                    polynomial(&f, &weight, bit)
                };
                tr.observe_fields(&p.0);
                let a = tr.sample();
                if first {
                    f = fold_table_base(&data.message, bit, a);
                } else {
                    fold_table(&mut f, bit, a);
                }
                fold_table(&mut weight, bit, a);
                alpha.push(a);
                polynomials.push(p);
            }
            let root = if i > 1 {
                let word = if trees.is_empty() {
                    self.code.fold_top(
                        i,
                        mmcs_base::<B>().get_matrices(&data.tree)[0],
                        &eq_weights(&alpha),
                    )
                } else {
                    self.code.fold(
                        i,
                        mmcs().get_matrices(trees.last().unwrap())[0],
                        &eq_weights(&alpha),
                    )
                };
                let (root, tree) = mmcs().commit_matrix(word);
                let root = root.roots()[0];
                tr.observe_bytes(&root);
                trees.push(tree);
                Some(root)
            } else {
                None
            };
            rounds.push(RoundProof { polynomials, root });
        }
        tr.observe_fields(&f);
        let mut queries = vec![];
        for _ in 0..c.queries {
            let mut j = tr.index(c.n[c.d - 1]);
            let opening = mmcs_base::<B>().open_batch(j, &data.tree);
            let top = TopOpening {
                values: opening.opened_values[0].clone(),
                path: opening.opening_proof,
            };
            let mut rest = vec![];
            for i in (1..c.d).rev() {
                let omitted = j / c.n[i - 1];
                j %= c.n[i - 1];
                let opening = mmcs().open_batch(j, &trees[c.d - i - 1]);
                let mut row = opening.opened_values[0].clone();
                let _ = row.remove(omitted);
                rest.push(QueryOpening {
                    values: row,
                    path: opening.opening_proof,
                });
            }
            queries.push(QueryNative { top, rest });
        }
        NativeProof {
            rounds,
            final_table: f,
            queries,
        }
    }
    pub fn verify(
        &self,
        cm: &Commitment,
        point: Vec<BF128>,
        value: BF128,
        proof: &NativeProof<B>,
    ) -> Result<(), VerifierError> {
        if self.verify_inner(cm, &point, value, proof) {
            Ok(())
        } else {
            Err(VerifierError::InvalidProof)
        }
    }
    fn verify_inner(
        &self,
        cm: &Commitment,
        point: &[BF128],
        value: BF128,
        proof: &NativeProof<B>,
    ) -> bool {
        let c = &self.config;
        let r = c.params.r;
        if point.len() != c.params.mu
            || cm.z0.len() != c.params.mu
            || proof.rounds.len() != c.d
            || proof.final_table.len() != c.k[0]
            || proof.queries.len() != c.queries
        {
            return false;
        }
        for (index, round) in proof.rounds.iter().enumerate() {
            if round.polynomials.len() != c.nu || round.root.is_some() != (index + 1 < c.d) {
                return false;
            }
        }
        for q in &proof.queries {
            if q.top.values.len() != r
                || q.top.path.len() != c.n[c.d - 1].ilog2() as usize
                || q.rest.len() != c.d - 1
            {
                return false;
            }
            for (index, v) in q.rest.iter().enumerate() {
                if v.values.len() != r - 1 || v.path.len() != c.n[c.d - index - 2].ilog2() as usize
                {
                    return false;
                }
            }
        }
        let Some(mut tr) = self.claim_transcript(cm, point, value) else {
            return false;
        };
        let gamma = tr.sample();
        let mut sum = cm.y0 + gamma * value;
        let mut challenges = vec![vec![]; c.d];
        let mut roots = vec![cm.root];
        for (index, round) in proof.rounds.iter().enumerate() {
            let i = c.d - index;
            for &p in &round.polynomials {
                if p.sum() != sum {
                    return false;
                }
                tr.observe_fields(&p.0);
                let a = tr.sample();
                challenges[i - 1].push(a);
                sum = p.evaluate(a);
            }
            if let Some(root) = round.root {
                tr.observe_bytes(&root);
                roots.push(root);
            }
        }
        tr.observe_fields(&proof.final_table);
        let alpha: Vec<_> = challenges.iter().flatten().copied().collect();
        let factor = sumcheck::eq(&cm.z0[c.mu_0..], &alpha);
        let mut w = sumcheck::eq_table(&cm.z0[..c.mu_0]);
        for x in &mut w {
            *x *= factor;
        }
        let factor = gamma * sumcheck::eq(&point[c.mu_0..], &alpha);
        for (x, y) in w.iter_mut().zip(sumcheck::eq_table(&point[..c.mu_0])) {
            *x += factor * y;
        }
        if sum
            != proof
                .final_table
                .iter()
                .zip(w)
                .map(|(&f, w)| f * w)
                .sum::<BF128>()
        {
            return false;
        }
        let base = base_code::encode(&proof.final_table, c.params.c.ilog2() as usize);
        let betas: Vec<_> = challenges.iter().map(|a| eq_weights(a)).collect();
        for q in &proof.queries {
            let mut j = tr.index(c.n[c.d - 1]);
            let rows = vec![q.top.values.clone()];
            if mmcs_base::<B>()
                .verify_batch(
                    &cap_base::<B>(cm.root),
                    &[Dimensions {
                        width: r,
                        height: c.n[c.d - 1],
                    }],
                    j,
                    BatchOpeningRef::new(&rows, &q.top.path),
                )
                .is_err()
            {
                return false;
            }
            let mut folded = self.code.fold_row_top(c.d, j, &rows[0], &betas[c.d - 1]);
            for (index, v) in q.rest.iter().enumerate() {
                let i = c.d - index - 1;
                let col = j / c.n[i - 1];
                j %= c.n[i - 1];
                let mut row = v.values.clone();
                row.insert(col, folded);
                let rows = vec![row];
                if mmcs()
                    .verify_batch(
                        &cap(roots[index + 1]),
                        &[Dimensions {
                            width: r,
                            height: c.n[i - 1],
                        }],
                        j,
                        BatchOpeningRef::new(&rows, &v.path),
                    )
                    .is_err()
                {
                    return false;
                }
                folded = self.code.fold_row(i, j, &rows[0], &betas[i - 1]);
            }
            if folded != base[j] {
                return false;
            }
        }
        true
    }
}

pub type BinaryHyperFoldPcs16 = NativePcs<BF16>;
pub type BinaryHyperFoldPcs32 = NativePcs<BF32>;
pub type BinaryHyperFoldPcs64 = NativePcs<BF64>;
pub type Proof64 = NativeProof<BF64>;
pub type ProverData64 = ProverDataNative<BF64>;

pub type Tree64 = TreeBase<BF64>;
pub type TopOpening64 = TopOpening<BF64>;
pub type Query64 = QueryNative<BF64>;
