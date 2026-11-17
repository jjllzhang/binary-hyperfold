//! Banked single-point opener. Sumcheck runs on low-coordinate projections
//! of the message, while oracle folds and the transcript remain unchanged.
use super::{
    BinaryHyperFoldPcs, ProverData, Tree, commit_folded, mmcs,
    proof::{HyperFoldProof, RoundProof},
};
use crate::{
    BF128,
    code::folding::eq_weights,
    sumcheck::{SumcheckPolynomial, dense},
};
use p3_binary_field::Ghash128 as G;
use p3_commit::Mmcs;
use p3_field::PrimeCharacteristicRing;
use p3_maybe_rayon::prelude::*;

// For each upper-bit assignment u, f[j] = <message[u, *], eq(point_j_low, *)>
// and w[j] = gamma_j * eq(point_j_high, u). Binding upper bits commutes with
// this low-bit projection, so the usual two-coefficient sumcheck runs on these
// short tables without constructing the full combined N-element weight.
#[derive(Clone, Copy)]
struct BankRow {
    f: [G; 2],
    w: [G; 2],
}

struct BankedProver<'a> {
    message: &'a [BF128],
    low_bits: usize,
    rows: Vec<BankRow>,
    challenges: Vec<BF128>,
}
impl<'a> BankedProver<'a> {
    fn new(
        message: &'a [BF128],
        z0: &[BF128],
        point: &[BF128],
        gamma: BF128,
        low_bits: usize,
    ) -> Self {
        let low_len = 1 << low_bits;
        let lo0 = dense::combined_weights(&[(&z0[..low_bits], BF128::ONE)]);
        let lo1 = dense::combined_weights(&[(&point[..low_bits], BF128::ONE)]);
        let hi0 = dense::combined_weights(&[(&z0[low_bits..], BF128::ONE)]);
        let hi1 = dense::combined_weights(&[(&point[low_bits..], gamma)]);
        let rows = message
            .par_chunks(low_len)
            .enumerate()
            .map(|(u, input)| {
                let mut f = [G::ZERO; 2];
                for (k, &value) in input.iter().enumerate() {
                    let value = G::from(value);
                    f[0] += value * lo0[k];
                    f[1] += value * lo1[k];
                }
                BankRow {
                    f,
                    w: [hi0[u], hi1[u]],
                }
            })
            .collect();
        Self {
            message,
            low_bits,
            rows,
            challenges: vec![BF128::ZERO; z0.len() - low_bits],
        }
    }

    fn polynomial(&self, bit: usize, running: BF128) -> SumcheckPolynomial {
        let bit = bit - self.low_bits;
        let stride = 1 << bit;
        let half = self.rows.len() / 2;
        let sums = (0..half).into_par_iter().par_fold_reduce(
            || [G::ZERO; 2],
            |mut sums, pair| {
                let j = ((pair >> bit) << (bit + 1)) + (pair & (stride - 1));
                let x = self.rows[j];
                let y = self.rows[j + stride];
                sums[0] += G::dot_product::<2>(&x.f, &x.w);
                sums[1] += G::dot_product::<2>(
                    &[x.f[0] + y.f[0], x.f[1] + y.f[1]],
                    &[x.w[0] + y.w[0], x.w[1] + y.w[1]],
                );
                sums
            },
            |a, b| [a[0] + b[0], a[1] + b[1]],
        );
        let c2 = BF128::from(sums[1]);
        SumcheckPolynomial([BF128::from(sums[0]), running + c2, c2])
    }

    fn bind(&mut self, bit: usize, step: usize, alpha: BF128) {
        self.challenges[bit + step - self.low_bits] = alpha;
        let bit = bit - self.low_bits;
        let stride = 1 << bit;
        let a = G::from(alpha);
        self.rows = (0..self.rows.len() / 2)
            .into_par_iter()
            .map(|pair| {
                let j = ((pair >> bit) << (bit + 1)) + (pair & (stride - 1));
                let x = self.rows[j];
                let y = self.rows[j + stride];
                BankRow {
                    f: core::array::from_fn(|k| x.f[k] + a * (x.f[k] + y.f[k])),
                    w: core::array::from_fn(|k| x.w[k] + a * (x.w[k] + y.w[k])),
                }
            })
            .collect();
    }

    // The protocol still sends the actual low-bit message table, not a bank.
    // Evaluate the original message at all sampled upper challenges in one
    // row-major scan, retaining the same canonical tower-basis wire encoding.
    fn final_table(&self) -> Vec<BF128> {
        let low_len = 1 << self.low_bits;
        let upper = dense::combined_weights(&[(&self.challenges, BF128::ONE)]);
        let result = self
            .message
            .par_chunks(low_len)
            .enumerate()
            .par_fold_reduce(
                || vec![G::ZERO; low_len],
                |mut partial, (row, input)| {
                    for (out, &value) in partial.iter_mut().zip(input) {
                        *out += G::from(value) * upper[row];
                    }
                    partial
                },
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            );
        result.into_iter().map(BF128::from).collect()
    }
}

impl BinaryHyperFoldPcs {
    /// Single-point opener without a full combined-weight table. The proof and
    /// transcript are identical to `open`; useful primarily for larger tables.
    pub fn open_banked(
        &self,
        data: &ProverData,
        point: Vec<BF128>,
        value: BF128,
    ) -> HyperFoldProof {
        assert_eq!(point.len(), self.config.params.mu);
        debug_assert_eq!(crate::sumcheck::evaluate(&data.message, &point), value);
        let mut tr = self
            .claim_transcript(&data.commitment, std::slice::from_ref(&point), &[value])
            .unwrap();
        let gamma = tr.sample();
        let mut running = data.commitment.y0 + gamma * value;
        let mut prover = BankedProver::new(
            &data.message,
            &data.commitment.z0,
            &point,
            gamma,
            self.config.mu_0,
        );
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
                prover.bind(bit, step, a);
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
        self.finish_opening(data, tr, rounds, &trees, prover.final_table(), 1)
    }
}
