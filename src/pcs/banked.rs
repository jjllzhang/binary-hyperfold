//! Banked opener. Sumcheck runs on low-coordinate projections
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
struct BankRow<const K: usize> {
    f: [G; K],
    w: [G; K],
}

struct BankedProver<'a, const K: usize> {
    message: &'a [BF128],
    low_bits: usize,
    rows: Vec<BankRow<K>>,
    challenges: Vec<BF128>,
}
impl<'a, const K: usize> BankedProver<'a, K> {
    fn new(message: &'a [BF128], terms: &[(&[BF128], BF128)], low_bits: usize) -> Self {
        assert!(!terms.is_empty() && terms.len() <= K);
        let low_len = 1 << low_bits;
        let (lows, highs): (Vec<_>, Vec<_>) = terms
            .iter()
            .map(|&(point, power)| {
                let low = dense::combined_weights(&[(&point[..low_bits], BF128::ONE)]);
                let high = dense::combined_weights(&[(&point[low_bits..], power)]);
                (low, high)
            })
            .unzip();
        let rows = message
            .par_chunks(low_len)
            .enumerate()
            .map(|(u, input)| {
                let mut f = [G::ZERO; K];
                for (k, &value) in input.iter().enumerate() {
                    let value = G::from(value);
                    for (bank, low) in f.iter_mut().zip(&lows) {
                        *bank += value * low[k];
                    }
                }
                BankRow {
                    f,
                    w: core::array::from_fn(|j| highs.get(j).map_or(G::ZERO, |high| high[u])),
                }
            })
            .collect();
        Self {
            message,
            low_bits,
            rows,
            challenges: vec![BF128::ZERO; terms[0].0.len() - low_bits],
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
                sums[0] += G::dot_product::<K>(&x.f, &x.w);
                sums[1] += G::dot_product::<K>(
                    &core::array::from_fn(|k| x.f[k] + y.f[k]),
                    &core::array::from_fn(|k| x.w[k] + y.w[k]),
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

// Keep common batch sizes in inline fixed-size rows. For larger batches,
// split the claims into banks of 33: every claim still participates in one
// combined sumcheck, without imposing a limit on the number of points.
enum BankedBatch<'a> {
    Two(BankedProver<'a, 2>),
    Three(BankedProver<'a, 3>),
    Five(BankedProver<'a, 5>),
    Nine(BankedProver<'a, 9>),
    Seventeen(BankedProver<'a, 17>),
    ThirtyThree(BankedProver<'a, 33>),
    Many(Vec<BankedProver<'a, 33>>),
}

impl<'a> BankedBatch<'a> {
    fn new(message: &'a [BF128], terms: &[(&[BF128], BF128)], low_bits: usize) -> Self {
        match terms.len() {
            2 => Self::Two(BankedProver::new(message, terms, low_bits)),
            3 => Self::Three(BankedProver::new(message, terms, low_bits)),
            4..=5 => Self::Five(BankedProver::new(message, terms, low_bits)),
            6..=9 => Self::Nine(BankedProver::new(message, terms, low_bits)),
            10..=17 => Self::Seventeen(BankedProver::new(message, terms, low_bits)),
            18..=33 => Self::ThirtyThree(BankedProver::new(message, terms, low_bits)),
            _ => Self::Many(
                terms
                    .chunks(33)
                    .map(|chunk| BankedProver::new(message, chunk, low_bits))
                    .collect(),
            ),
        }
    }

    fn polynomial(&self, bit: usize, running: BF128) -> SumcheckPolynomial {
        match self {
            Self::Two(p) => p.polynomial(bit, running),
            Self::Three(p) => p.polynomial(bit, running),
            Self::Five(p) => p.polynomial(bit, running),
            Self::Nine(p) => p.polynomial(bit, running),
            Self::Seventeen(p) => p.polynomial(bit, running),
            Self::ThirtyThree(p) => p.polynomial(bit, running),
            Self::Many(banks) => {
                let (c0, c2) = banks
                    .iter()
                    .fold((BF128::ZERO, BF128::ZERO), |(c0, c2), p| {
                        let polynomial = p.polynomial(bit, BF128::ZERO);
                        (c0 + polynomial.0[0], c2 + polynomial.0[2])
                    });
                SumcheckPolynomial([c0, running + c2, c2])
            }
        }
    }

    fn bind(&mut self, bit: usize, step: usize, alpha: BF128) {
        match self {
            Self::Two(p) => p.bind(bit, step, alpha),
            Self::Three(p) => p.bind(bit, step, alpha),
            Self::Five(p) => p.bind(bit, step, alpha),
            Self::Nine(p) => p.bind(bit, step, alpha),
            Self::Seventeen(p) => p.bind(bit, step, alpha),
            Self::ThirtyThree(p) => p.bind(bit, step, alpha),
            Self::Many(banks) => {
                for p in banks {
                    p.bind(bit, step, alpha);
                }
            }
        }
    }

    fn final_table(&self) -> Vec<BF128> {
        match self {
            Self::Two(p) => p.final_table(),
            Self::Three(p) => p.final_table(),
            Self::Five(p) => p.final_table(),
            Self::Nine(p) => p.final_table(),
            Self::Seventeen(p) => p.final_table(),
            Self::ThirtyThree(p) => p.final_table(),
            Self::Many(banks) => banks[0].final_table(),
        }
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
        self.open_banked_batched(data, &[point], &[value])
    }

    pub(super) fn open_banked_batched(
        &self,
        data: &ProverData,
        points: &[Vec<BF128>],
        values: &[BF128],
    ) -> HyperFoldProof {
        let mut tr = self
            .claim_transcript(&data.commitment, points, values)
            .unwrap();
        let gamma = tr.sample();
        let mut terms = vec![(data.commitment.z0.as_slice(), BF128::ONE)];
        let mut running = data.commitment.y0;
        let mut power = gamma;
        for (point, &value) in points.iter().zip(values) {
            debug_assert_eq!(crate::sumcheck::evaluate(&data.message, point), value);
            terms.push((point.as_slice(), power));
            running += power * value;
            power *= gamma;
        }
        let mut prover = BankedBatch::new(&data.message, &terms, self.config.mu_0);
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
        self.finish_opening(data, tr, rounds, &trees, prover.final_table(), points.len())
    }
}
