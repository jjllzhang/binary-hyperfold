use super::{
    BinaryHyperFoldPcs, Commitment, ProverData, Tree, commit_encoded, mmcs,
    multiproof::layer_indices,
    proof::{HyperFoldProof, QueryOpening, RoundProof},
};
use crate::{BF128, sumcheck, transcript::ProverTranscript};
use p3_commit::Mmcs;

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
        // Single- and multi-point openings always use banked sumcheck.
        self.open_banked_batched(data, points, values)
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
