use super::{
    BinaryHyperFoldPcs, Commitment, cap, error::VerifierError, mmcs, proof::HyperFoldProof,
};
use crate::{
    BF128,
    code::{base_code, folding::eq_weights},
    sumcheck,
};
use p3_commit::{BatchOpeningRef, Mmcs};

use p3_matrix::Dimensions;
impl BinaryHyperFoldPcs {
    pub fn verify(
        &self,
        cm: &Commitment,
        point: Vec<BF128>,
        value: BF128,
        proof: &HyperFoldProof,
    ) -> Result<(), VerifierError> {
        self.verify_batched(cm, &[point], &[value], proof)
    }
    pub fn verify_batched(
        &self,
        cm: &Commitment,
        points: &[Vec<BF128>],
        values: &[BF128],
        proof: &HyperFoldProof,
    ) -> Result<(), VerifierError> {
        if self.verify_inner(cm, points, values, proof) {
            Ok(())
        } else {
            Err(VerifierError::InvalidProof)
        }
    }
    fn verify_inner(
        &self,
        cm: &Commitment,
        points: &[Vec<BF128>],
        values: &[BF128],
        proof: &HyperFoldProof,
    ) -> bool {
        let c = &self.config;
        let r = c.params.r;
        if points.is_empty()
            || points.len() != values.len()
            || points.iter().any(|p| p.len() != c.params.mu)
            || cm.z0.len() != c.params.mu
            || proof.rounds.len() != c.d
            || proof.final_table.len() != c.k[0]
            || proof.queries.len() != self.query_count(points.len())
        {
            return false;
        }
        for (j, round) in proof.rounds.iter().enumerate() {
            if round.polynomials.len() != c.nu || round.root.is_some() != (j + 1 < c.d) {
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
        let Some(mut tr) = self.claim_transcript(cm, points, values) else {
            return false;
        };
        let gamma = tr.sample();
        let mut power = gamma;
        let mut sum = cm.y0;
        for &v in values {
            sum += power * v;
            power *= gamma
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
                sum = p.evaluate(a)
            }
            if let Some(root) = round.root {
                tr.observe_bytes(&root);
                roots.push(root)
            }
        }
        tr.observe_fields(&proof.final_table);
        let alpha: Vec<_> = challenges.iter().flatten().copied().collect();
        let mut w = sumcheck::eq_table(&cm.z0[..c.mu_0]);
        let factor = sumcheck::eq(&cm.z0[c.mu_0..], &alpha);
        for x in &mut w {
            *x *= factor
        }
        power = gamma;
        for point in points {
            let f = power * sumcheck::eq(&point[c.mu_0..], &alpha);
            for (x, y) in w.iter_mut().zip(sumcheck::eq_table(&point[..c.mu_0])) {
                *x += f * y
            }
            power *= gamma;
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
        for query in &proof.queries {
            let mut j = tr.index(c.n[c.d - 1]);
            let mut prev = None;
            for (index, opening) in query.iter().enumerate() {
                let i = c.d - index;
                let mut row = opening.values.clone();
                if let Some((col, value)) = prev {
                    row.insert(col, value)
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
                let folded = self
                    .code
                    .fold_row(i, j, &rows[0], &eq_weights(&challenges[i - 1]));
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
