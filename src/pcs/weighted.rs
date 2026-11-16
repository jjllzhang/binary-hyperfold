//! Transparent linear-functional openings. The original point-opening protocol is unchanged.
use super::{BinaryHyperFoldPcs, Commitment, ProverData, Tree, cap, error::VerifierError, mmcs};
use crate::{
    BF128,
    code::{base_code, folding::eq_weights},
    sumcheck::{
        self, SumcheckPolynomial,
        prover::{fold_table, polynomial},
    },
    transcript::ProverTranscript,
};
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_matrix::Dimensions;
use serde::{Deserialize, Serialize};

/// Public, transcript-bound description of a multilinear weight. The prover
/// also supplies its Boolean table; the verifier needs only succinct evaluation.
/// The caller must construct both views from the same verified public statement.
pub trait TransparentWeight {
    fn table(&self) -> Option<&[BF128]>;
    fn evaluate(&self, point: &[BF128]) -> BF128;
    fn observe(&self, transcript: &mut ProverTranscript);
}

pub struct WeightedClaim<'a> {
    pub weight: &'a dyn TransparentWeight,
    pub value: BF128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightedHyperFoldProof {
    pub rounds: Vec<super::proof::RoundProof>,
    pub final_table: Vec<BF128>,
    pub base_polynomials: Vec<SumcheckPolynomial>,
    pub queries: Vec<Vec<super::proof::QueryOpening>>,
}
impl WeightedHyperFoldProof {
    pub fn size_in_bytes(&self) -> usize {
        self.rounds
            .iter()
            .map(|r| 48 * r.polynomials.len() + 32 * usize::from(r.root.is_some()))
            .sum::<usize>()
            + 16 * (self.final_table.len() + 3 * self.base_polynomials.len())
            + self
                .queries
                .iter()
                .flatten()
                .map(|q| 16 * q.values.len() + 32 * q.path.len())
                .sum::<usize>()
    }
    pub fn wire_size_in_bytes(&self) -> usize {
        bincode::serialized_size(self).unwrap() as usize
    }
}

impl BinaryHyperFoldPcs {
    fn weighted_transcript(
        &self,
        cm: &Commitment,
        claims: &[WeightedClaim<'_>],
    ) -> Option<ProverTranscript> {
        let mut tr = self.commitment_transcript(cm)?;
        tr.observe_bytes(b"binary-hyperfold-weighted-v1");
        tr.observe_bytes(&(claims.len() as u64).to_le_bytes());
        for claim in claims {
            claim.weight.observe(&mut tr);
            tr.observe(claim.value);
        }
        Some(tr)
    }
    fn weighted_queries(&self, m: usize) -> usize {
        let c = &self.config;
        let extra = ((m.saturating_sub(1) * c.list_size + 2 * c.mu_0) as f64) / 2f64.powi(128);
        crate::parameters::soundness::query_count(
            c.params.security_bits,
            c.epsilon_code,
            c.epsilon_field + extra,
            c.delta,
        )
        .expect("no weighted soundness budget")
    }

    /// Prove any fixed set of transparent inner products of the committed table.
    /// The original point-valued `open` and `open_batched` remain separate.
    pub fn open_weighted(
        &self,
        data: &ProverData,
        claims: &[WeightedClaim<'_>],
    ) -> WeightedHyperFoldProof {
        assert!(!claims.is_empty());
        let mut tr = self
            .weighted_transcript(&data.commitment, claims)
            .expect("commitment mismatch");
        let gamma = tr.sample();
        let mut weight = sumcheck::eq_table(&data.commitment.z0);
        let mut power = gamma;
        for claim in claims {
            let table = claim
                .weight
                .table()
                .expect("prover needs the transparent weight table");
            assert_eq!(table.len(), data.message.len());
            debug_assert_eq!(
                data.message
                    .iter()
                    .zip(table)
                    .map(|(&a, &b)| a * b)
                    .sum::<BF128>(),
                claim.value
            );
            for (w, &b) in weight.iter_mut().zip(table) {
                *w += power * b;
            }
            power *= gamma;
        }
        let mut f = data.message.clone();
        let mut rounds = vec![];
        let mut trees: Vec<Tree> = vec![];
        for i in (1..=self.config.d).rev() {
            let bit = self.config.mu_0 + (i - 1) * self.config.nu;
            let mut polynomials = vec![];
            let mut alpha = vec![];
            for _ in 0..self.config.nu {
                let p = polynomial(&f, &weight, bit);
                tr.observe_fields(&p.0);
                let a = tr.sample();
                fold_table(&mut f, bit, a);
                fold_table(&mut weight, bit, a);
                alpha.push(a);
                polynomials.push(p);
            }
            let root = if i > 1 {
                let source = if trees.is_empty() {
                    &data.tree
                } else {
                    trees.last().unwrap()
                };
                let matrix = mmcs().get_matrices(source)[0];
                let (root, tree) =
                    mmcs().commit_matrix(self.code.fold(i, matrix, &eq_weights(&alpha)));
                let root = root.roots()[0];
                tr.observe_bytes(&root);
                trees.push(tree);
                Some(root)
            } else {
                None
            };
            rounds.push(super::proof::RoundProof { polynomials, root });
        }
        let final_table = f.clone();
        tr.observe_fields(&final_table);
        let mut base_polynomials = Vec::with_capacity(self.config.mu_0);
        for _ in 0..self.config.mu_0 {
            let p = polynomial(&f, &weight, 0);
            tr.observe_fields(&p.0);
            let a = tr.sample();
            fold_table(&mut f, 0, a);
            fold_table(&mut weight, 0, a);
            base_polynomials.push(p);
        }
        let mut queries = vec![];
        for _ in 0..self.weighted_queries(claims.len()) {
            let mut j = tr.index(self.config.n[self.config.d - 1]);
            let mut layers = vec![];
            let mut omitted = None;
            for i in (1..=self.config.d).rev() {
                let tree = if i == self.config.d {
                    &data.tree
                } else {
                    &trees[self.config.d - i - 1]
                };
                let opening = mmcs().open_batch(j, tree);
                let mut row = opening.opened_values[0].clone();
                if let Some(col) = omitted {
                    let _ = row.remove(col);
                }
                layers.push(super::proof::QueryOpening {
                    values: row,
                    path: opening.opening_proof,
                });
                if i > 1 {
                    omitted = Some(j / self.config.n[i - 2]);
                    j %= self.config.n[i - 2];
                }
            }
            queries.push(layers);
        }
        WeightedHyperFoldProof {
            rounds,
            final_table,
            base_polynomials,
            queries,
        }
    }

    pub fn verify_weighted(
        &self,
        cm: &Commitment,
        claims: &[WeightedClaim<'_>],
        proof: &WeightedHyperFoldProof,
    ) -> Result<(), VerifierError> {
        if self.verify_weighted_inner(cm, claims, proof) {
            Ok(())
        } else {
            Err(VerifierError::InvalidProof)
        }
    }
    fn verify_weighted_inner(
        &self,
        cm: &Commitment,
        claims: &[WeightedClaim<'_>],
        proof: &WeightedHyperFoldProof,
    ) -> bool {
        let c = &self.config;
        let r = c.params.r;
        if claims.is_empty()
            || cm.z0.len() != c.params.mu
            || proof.rounds.len() != c.d
            || proof.final_table.len() != c.k[0]
            || proof.base_polynomials.len() != c.mu_0
            || proof.queries.len() != self.weighted_queries(claims.len())
        {
            return false;
        }
        for (index, round) in proof.rounds.iter().enumerate() {
            if round.polynomials.len() != c.nu || round.root.is_some() != (index + 1 < c.d) {
                return false;
            }
        }
        for query in &proof.queries {
            if query.len() != c.d {
                return false;
            }
            for (index, row) in query.iter().enumerate() {
                let i = c.d - index;
                if row.values.len() != if index == 0 { r } else { r - 1 }
                    || row.path.len() != c.n[i - 1].ilog2() as usize
                {
                    return false;
                }
            }
        }
        let Some(mut tr) = self.weighted_transcript(cm, claims) else {
            return false;
        };
        let gamma = tr.sample();
        let mut power = gamma;
        let mut sum = cm.y0;
        for claim in claims {
            sum += power * claim.value;
            power *= gamma;
        }
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
        let mut bottom_point = Vec::with_capacity(c.params.mu);
        for &p in &proof.base_polynomials {
            if p.sum() != sum {
                return false;
            }
            tr.observe_fields(&p.0);
            let a = tr.sample();
            bottom_point.push(a);
            sum = p.evaluate(a);
        }
        let f_at_point = sumcheck::evaluate(&proof.final_table, &bottom_point);
        let mut full_point = bottom_point;
        full_point.extend(challenges.iter().flatten().copied());
        let mut w_at_point = sumcheck::eq(&cm.z0, &full_point);
        power = gamma;
        for claim in claims {
            w_at_point += power * claim.weight.evaluate(&full_point);
            power *= gamma;
        }
        if sum != f_at_point * w_at_point {
            return false;
        }
        let base = base_code::encode(&proof.final_table, c.params.c.ilog2() as usize);
        let betas: Vec<_> = challenges.iter().map(|alpha| eq_weights(alpha)).collect();
        for query in &proof.queries {
            let mut j = tr.index(c.n[c.d - 1]);
            let mut prev = None;
            for (index, opening) in query.iter().enumerate() {
                let i = c.d - index;
                let mut row = opening.values.clone();
                if let Some((col, value)) = prev {
                    row.insert(col, value);
                }
                let rows = vec![row];
                if mmcs()
                    .verify_batch(
                        &cap(roots[index]),
                        &[Dimensions {
                            width: r,
                            height: c.n[i - 1],
                        }],
                        j,
                        BatchOpeningRef::new(&rows, &opening.path),
                    )
                    .is_err()
                {
                    return false;
                }
                let folded = self.code.fold_row(i, j, &rows[0], &betas[i - 1]);
                if i == 1 {
                    if folded != base[j] {
                        return false;
                    }
                } else {
                    prev = Some((j / c.n[i - 2], folded));
                    j %= c.n[i - 2];
                }
            }
        }
        true
    }
}
