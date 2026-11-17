//! Transparent linear-functional openings. The original point-opening protocol is unchanged.
use super::{
    BinaryHyperFoldPcs, Commitment, ProverData, Tree, cap, commit_folded,
    error::VerifierError,
    mmcs,
    multiproof::{MultiPath, layer_indices, path_bytes},
};
use crate::{
    BF128,
    code::{base_code, folding::eq_weights},
    sumcheck::{
        self, SumcheckPolynomial,
        dense::{self, DenseProver},
    },
    transcript::ProverTranscript,
};
use p3_commit::Mmcs;
use p3_field::PrimeCharacteristicRing;
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
    pub paths: Vec<MultiPath>,
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
                .map(|q| 16 * q.values.len())
                .sum::<usize>()
            + path_bytes(&self.paths)
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
        let mut weight = dense::combined_weights(&[(&data.commitment.z0, BF128::ONE)]);
        let mut running = data.commitment.y0;
        let mut terms = Vec::with_capacity(claims.len());
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
            terms.push((table, power));
            running += power * claim.value;
            power *= gamma;
        }
        dense::add_scaled_tables(&mut weight, &terms);
        let mut prover = DenseProver::new(&data.message, weight);
        let mut rounds = vec![];
        let mut trees: Vec<Tree> = vec![];
        for i in (1..=self.config.d).rev() {
            let bit = self.config.mu_0 + (i - 1) * self.config.nu;
            let mut polynomials = vec![];
            let mut alpha = vec![];
            for step in 0..self.config.nu {
                let p = prover.polynomial(bit, running);
                tr.observe_fields(&p.0);
                let a = tr.sample();
                running = p.evaluate(a);
                let next = if step + 1 < self.config.nu {
                    Some(bit)
                } else if i > 1 {
                    Some(bit - self.config.nu)
                } else if self.config.mu_0 > 0 {
                    Some(0)
                } else {
                    None
                };
                prover.bind(bit, a, next);
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
                let (root, tree) = commit_folded(&self.code, i, matrix, &eq_weights(&alpha));
                let root = root.roots()[0];
                tr.observe_bytes(&root);
                trees.push(tree);
                Some(root)
            } else {
                None
            };
            rounds.push(super::proof::RoundProof { polynomials, root });
        }
        let final_table = prover.table();
        tr.observe_fields(&final_table);
        let mut base_polynomials = Vec::with_capacity(self.config.mu_0);
        for step in 0..self.config.mu_0 {
            let p = prover.polynomial(0, running);
            tr.observe_fields(&p.0);
            let a = tr.sample();
            running = p.evaluate(a);
            prover.bind(0, a, (step + 1 < self.config.mu_0).then_some(0));
            base_polynomials.push(p);
        }
        let indices = layer_indices(
            &mut tr,
            &self.config.n[..self.config.d],
            self.weighted_queries(claims.len()),
        );
        let mut queries = vec![vec![]; indices[0].len()];
        let mut paths = vec![];
        for (layer, positions) in indices.iter().enumerate() {
            let tree = if layer == 0 {
                &data.tree
            } else {
                &trees[layer - 1]
            };
            let (rows, path) = mmcs().open_multi_batch(positions, tree);
            paths.push(path);
            for (q, values) in rows.into_iter().enumerate() {
                let mut row = values.into_iter().next().unwrap();
                if layer > 0 {
                    let _ = row
                        .remove(indices[layer - 1][q] / self.config.n[self.config.d - layer - 1]);
                }
                queries[q].push(super::proof::QueryOpening { values: row });
            }
        }
        WeightedHyperFoldProof {
            rounds,
            final_table,
            base_polynomials,
            queries,
            paths,
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
            || proof.paths.len() != c.d
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
                if row.values.len() != if index == 0 { r } else { r - 1 } {
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
        let indices = layer_indices(&mut tr, &c.n[..c.d], proof.queries.len());
        let mut prev = vec![BF128::default(); proof.queries.len()];
        for (layer, positions) in indices.iter().enumerate() {
            let i = c.d - layer;
            let rows: Vec<Vec<BF128>> = proof
                .queries
                .iter()
                .enumerate()
                .map(|(q, query)| {
                    let mut row = query[layer].values.clone();
                    if layer > 0 {
                        row.insert(indices[layer - 1][q] / c.n[i - 1], prev[q]);
                    }
                    row
                })
                .collect();
            let opened: Vec<_> = rows.iter().cloned().map(|row| vec![row]).collect();
            if mmcs()
                .verify_multi_batch(
                    &cap(roots[layer]),
                    &[Dimensions {
                        width: r,
                        height: c.n[i - 1],
                    }],
                    positions,
                    &opened,
                    &proof.paths[layer],
                )
                .is_err()
            {
                return false;
            }
            for (q, row) in rows.iter().enumerate() {
                let folded = self.code.fold_row(i, positions[q], row, &betas[i - 1]);
                if i == 1 {
                    if folded != base[positions[q]] {
                        return false;
                    }
                } else {
                    prev[q] = folded;
                }
            }
        }
        true
    }
}
