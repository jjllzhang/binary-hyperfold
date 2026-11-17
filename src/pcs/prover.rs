use super::{
    BinaryHyperFoldPcs, Commitment, ProverData, Tree, commit_encoded, commit_folded, mmcs,
    multiproof::layer_indices,
    proof::{HyperFoldProof, QueryOpening, RoundProof},
};
use crate::{
    BF128,
    code::folding::eq_weights,
    sumcheck::{
        self,
        dense::{self, DenseProver},
    },
    transcript::ProverTranscript,
};
use p3_commit::Mmcs;
use p3_field::PrimeCharacteristicRing;

impl BinaryHyperFoldPcs {
    pub fn commit(&self, message: Vec<BF128>) -> (Commitment, ProverData) {
        let (root, tree) = commit_encoded(&self.code, self.config.d, &message);
        let root = root.roots()[0];
        let mut tr = self.transcript();
        tr.observe_bytes(&root);
        let z0 = tr.point(self.config.params.mu);
        let y0 = sumcheck::evaluate(&message, &z0);
        let cm = Commitment { root, z0, y0 };
        let data = ProverData {
            message,
            tree,
            commitment: cm.clone(),
        };
        (cm, data)
    }
    pub fn open(&self, data: &ProverData, point: Vec<BF128>, value: BF128) -> HyperFoldProof {
        self.open_batched(data, &[point], &[value])
    }
    pub fn open_batched(
        &self,
        data: &ProverData,
        points: &[Vec<BF128>],
        values: &[BF128],
    ) -> HyperFoldProof {
        assert!(!points.is_empty() && points.len() == values.len());
        assert!(points.iter().all(|p| p.len() == self.config.params.mu));
        // For the measured large, single-point configurations, keeping only
        // low-coordinate bank sums avoids the full GHASH weight table. Smaller
        // and batched openings retain the dense prover and its scheduling.
        if points.len() == 1
            && self.config.params.mu >= 22
            && self.config.params.c == 2
            && self.config.params.matrix_kind == crate::parameters::MatrixKind::Specialized
            && matches!(self.config.params.r, 4 | 8)
        {
            return self.open_banked(data, points[0].clone(), values[0]);
        }
        let mut tr = self
            .claim_transcript(&data.commitment, points, values)
            .unwrap();
        let gamma = tr.sample();
        let mut terms = vec![(data.commitment.z0.as_slice(), BF128::ONE)];
        let mut running = data.commitment.y0;
        let mut power = gamma;
        for (point, &value) in points.iter().zip(values) {
            debug_assert_eq!(sumcheck::evaluate(&data.message, point), value);
            terms.push((point.as_slice(), power));
            running += power * value;
            power *= gamma;
        }
        let mut prover = DenseProver::new(&data.message, dense::combined_weights(&terms));
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
            rounds.push(RoundProof { polynomials, root });
        }
        self.finish_opening(data, tr, rounds, &trees, prover.table(), points.len())
    }

    pub(super) fn finish_opening(
        &self,
        data: &ProverData,
        mut tr: ProverTranscript,
        rounds: Vec<RoundProof>,
        trees: &[Tree],
        f: Vec<BF128>,
        claims: usize,
    ) -> HyperFoldProof {
        tr.observe_fields(&f);
        let indices = layer_indices(
            &mut tr,
            &self.config.n[..self.config.d],
            self.query_count(claims),
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
                queries[q].push(QueryOpening { values: row });
            }
        }
        HyperFoldProof {
            rounds,
            final_table: f,
            queries,
            paths,
        }
    }
}
