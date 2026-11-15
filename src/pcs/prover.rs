use super::{
    BinaryHyperFoldPcs, Commitment, ProverData, Tree, mmcs,
    proof::{HyperFoldProof, QueryOpening, RoundProof},
};
use crate::{
    BF128,
    code::folding::eq_weights,
    sumcheck::{
        self,
        prover::{fold_table, polynomial},
    },
};
use p3_commit::Mmcs;

impl BinaryHyperFoldPcs {
    pub fn commit(&self, message: Vec<BF128>) -> (Commitment, ProverData) {
        let (root, tree) = mmcs().commit_matrix(self.code.encode(self.config.d, &message));
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
        let mut tr = self
            .claim_transcript(&data.commitment, points, values)
            .unwrap();
        let gamma = tr.sample();
        let mut weight = sumcheck::eq_table(&data.commitment.z0);
        let mut power = gamma;
        for (point, &value) in points.iter().zip(values) {
            debug_assert_eq!(sumcheck::evaluate(&data.message, point), value);
            let eq = sumcheck::eq_table(point);
            for (w, e) in weight.iter_mut().zip(eq) {
                *w += power * e
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
            rounds.push(RoundProof { polynomials, root });
        }
        tr.observe_fields(&f);
        let mut queries = vec![];
        for _ in 0..self.query_count(points.len()) {
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
                layers.push(QueryOpening {
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
        HyperFoldProof {
            rounds,
            final_table: f,
            queries,
        }
    }
}
