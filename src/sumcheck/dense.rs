//! Dense sumcheck arithmetic stays in the polynomial basis. Only protocol messages
//! and the final table cross back into the canonical tower representation.
use super::SumcheckPolynomial;
use crate::BF128;
use p3_binary_field::Ghash128 as G;
use p3_field::{ExtensionField, Field, PackedValue, PrimeCharacteristicRing};
use p3_maybe_rayon::prelude::*;

type P = <G as Field>::Packing;
const WIDTH: usize = P::WIDTH;
const CHUNK: usize = 1024;
const PARALLEL: usize = 4096;
// The upstream runtime GFNI map refuses batches shorter than 4096 elements.
const MAP_CHUNK: usize = 4096;

fn add(a: [G; 2], b: [G; 2]) -> [G; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

/// Accumulate four packed products before reducing each polynomial product sum.
struct Measurement {
    a: [P; 4],
    x: [P; 4],
    b: [P; 4],
    y: [P; 4],
    sums: [P; 2],
    used: usize,
}
impl Measurement {
    fn new() -> Self {
        Self {
            a: [P::ZERO; 4],
            x: [P::ZERO; 4],
            b: [P::ZERO; 4],
            y: [P::ZERO; 4],
            sums: [P::ZERO; 2],
            used: 0,
        }
    }
    #[inline]
    fn push(&mut self, a: P, b: P, x: P, y: P) {
        self.a[self.used] = a;
        self.b[self.used] = a + b;
        self.x[self.used] = x;
        self.y[self.used] = x + y;
        self.used += 1;
        if self.used == 4 {
            self.flush();
        }
    }
    #[inline]
    fn flush(&mut self) {
        self.sums[0] += P::dot_product(&self.a, &self.x);
        self.sums[1] += P::dot_product(&self.b, &self.y);
        self.used = 0;
    }
    fn finish(mut self) -> [G; 2] {
        if self.used != 0 {
            self.a[self.used..].fill(P::ZERO);
            self.b[self.used..].fill(P::ZERO);
            self.flush();
        }
        self.sums.map(|p| p.as_slice().iter().copied().sum())
    }
}

/// Map a compact pair index to its lower endpoint before binding `bit`.
#[inline]
fn source(pair: usize, bit: usize) -> usize {
    ((pair >> bit) << (bit + 1)) + (pair & ((1 << bit) - 1))
}

#[inline]
fn pair(table: &[G], start: usize, bit: usize) -> (P, P) {
    let stride = 1 << bit;
    if stride >= WIDTH {
        let j = source(start, bit);
        (
            *P::from_slice(&table[j..j + WIDTH]),
            *P::from_slice(&table[j + stride..j + stride + WIDTH]),
        )
    } else {
        (
            P::from_fn(|lane| table[source(start + lane, bit)]),
            P::from_fn(|lane| table[source(start + lane, bit) + stride]),
        )
    }
}

#[inline]
fn base_pair<B: Field>(table: &[B], start: usize, bit: usize) -> (P, P)
where
    BF128: ExtensionField<B>,
{
    let stride = 1 << bit;
    (
        P::from_fn(|lane| G::from(BF128::from(table[source(start + lane, bit)]))),
        P::from_fn(|lane| G::from(BF128::from(table[source(start + lane, bit) + stride]))),
    )
}

fn measure(len: usize, bit: usize, f: impl Fn(usize, usize) -> (P, P) + Sync, w: &[G]) -> [G; 2] {
    let half = len / 2;
    let block = |start: usize, end: usize| {
        let mut m = Measurement::new();
        let mut j = start;
        while j + WIDTH <= end {
            let (a, b) = f(j, bit);
            let (x, y) = pair(w, j, bit);
            m.push(a, b, x, y);
            j += WIDTH;
        }
        // Tables and task blocks have power-of-two lengths. Smaller-than-pack
        // tables take the scalar path before entering this kernel.
        debug_assert_eq!(j, end);
        m.finish()
    };
    if len < PARALLEL {
        block(0, half)
    } else {
        (0..half.div_ceil(CHUNK)).into_par_iter().par_fold_reduce(
            || [G::ZERO; 2],
            |sum, chunk| add(sum, block(chunk * CHUNK, ((chunk + 1) * CHUNK).min(half))),
            add,
        )
    }
}

fn factor(point: &[BF128], scale: G) -> Vec<G> {
    let mut out = vec![G::ZERO; 1 << point.len()];
    out[0] = scale;
    for (bit, &a) in point.iter().enumerate() {
        let a = G::from(a);
        let half = 1 << bit;
        let (lo, hi) = out[..2 * half].split_at_mut(half);
        for (x, y) in lo.iter_mut().zip(hi) {
            *y = *x * a;
            *x += *y;
        }
    }
    out
}

pub(crate) fn combined_weights(terms: &[(&[BF128], BF128)]) -> Vec<G> {
    assert!(!terms.is_empty());
    let variables = terms[0].0.len();
    assert!(terms.iter().all(|(point, _)| point.len() == variables));
    let split = variables / 2;
    let factors: Vec<_> = terms
        .iter()
        .map(|&(p, s)| (factor(&p[..split], G::ONE), factor(&p[split..], G::from(s))))
        .collect();
    let width = 1 << split;
    let mut out = vec![G::ZERO; 1 << variables];
    let fill = |j: usize, row: &mut [G]| {
        let mut k = 0;
        while k + WIDTH <= row.len() {
            let value: P = factors
                .iter()
                .map(|(lo, hi)| *P::from_slice(&lo[k..k + WIDTH]) * hi[j])
                .sum();
            *P::from_slice_mut(&mut row[k..k + WIDTH]) = value;
            k += WIDTH;
        }
        for (k, x) in row.iter_mut().enumerate().skip(k) {
            *x = factors.iter().map(|(lo, hi)| lo[k] * hi[j]).sum();
        }
    };
    if out.len() < PARALLEL {
        for (j, row) in out.chunks_mut(width).enumerate() {
            fill(j, row);
        }
    } else {
        out.par_chunks_mut(width)
            .enumerate()
            .for_each(|(j, row)| fill(j, row));
    }
    out
}

/// Convert and combine dense weights using a per-task scratch block, not a
/// second full-sized converted table. Multiplication by a fixed scale is F2-linear.
pub(crate) fn add_scaled_tables(weight: &mut [G], terms: &[(&[BF128], BF128)]) {
    let maps: Vec<_> = terms
        .iter()
        .map(|&(table, scale)| {
            assert_eq!(table.len(), weight.len());
            let scale = G::from(scale);
            let images = core::array::from_fn(|bit| {
                use p3_binary_field::TowerLevel;
                G::from(BF128::from_repr(1u128 << bit)) * scale
            });
            (table, scale, images)
        })
        .collect();
    let fill = |scratch: &mut Vec<G>, chunk: usize, block: &mut [G]| {
        let start = chunk * MAP_CHUNK;
        scratch.resize(block.len(), G::ZERO);
        for (table, scale, images) in &maps {
            let input = &table[start..start + block.len()];
            if !G::try_apply_linear_map_into(images, input, scratch) {
                for (out, &value) in scratch.iter_mut().zip(input) {
                    *out = G::from(value) * *scale;
                }
            }
            for (out, &value) in block.iter_mut().zip(scratch.iter()) {
                *out += value;
            }
        }
    };
    if weight.len() < PARALLEL {
        let mut scratch = Vec::new();
        for (chunk, block) in weight.chunks_mut(MAP_CHUNK).enumerate() {
            fill(&mut scratch, chunk, block);
        }
    } else {
        weight.par_chunks_mut(MAP_CHUNK).enumerate().for_each_init(
            || Vec::with_capacity(MAP_CHUNK),
            |scratch, (chunk, block)| fill(scratch, chunk, block),
        );
    }
}

pub(crate) struct DenseProver<'a> {
    initial: Option<&'a [BF128]>,
    f: Vec<G>,
    w: Vec<G>,
    f_scratch: Vec<G>,
    w_scratch: Vec<G>,
    measured: Option<(usize, [G; 2])>,
}
impl<'a> DenseProver<'a> {
    pub fn new(f: &'a [BF128], w: Vec<G>) -> Self {
        assert_eq!(f.len(), w.len());
        Self {
            initial: Some(f),
            ..Self::native(w)
        }
    }
    /// Native input is borrowed until its first binding, avoiding an N-element
    /// BF128 promotion. Only the half-size bound table is retained.
    pub fn native(w: Vec<G>) -> Self {
        Self {
            initial: None,
            f: Vec::new(),
            w,
            f_scratch: Vec::new(),
            w_scratch: Vec::new(),
            measured: None,
        }
    }
    fn message(
        &mut self,
        bit: usize,
        sum: BF128,
        measure: impl FnOnce(&[G]) -> [G; 2],
    ) -> SumcheckPolynomial {
        let c = if let Some((cached_bit, c)) = self.measured.take() {
            assert_eq!(cached_bit, bit);
            c
        } else {
            measure(&self.w)
        };
        SumcheckPolynomial([
            BF128::from(c[0]),
            sum + BF128::from(c[1]),
            BF128::from(c[1]),
        ])
    }
    pub fn polynomial(&mut self, bit: usize, sum: BF128) -> SumcheckPolynomial {
        if let Some(f) = self.initial {
            return self.polynomial_base(f, bit, sum);
        }
        let f = &self.f;
        let c = if self.measured.is_none() {
            Some(if f.len() / 2 < WIDTH {
                scalar_measure(f.len(), bit, |j| f[j], &self.w)
            } else {
                measure(f.len(), bit, |j, b| pair(f, j, b), &self.w)
            })
        } else {
            None
        };
        self.message(bit, sum, |_| c.unwrap())
    }
    pub fn polynomial_base<B: Field>(
        &mut self,
        f: &[B],
        bit: usize,
        sum: BF128,
    ) -> SumcheckPolynomial
    where
        BF128: ExtensionField<B>,
    {
        assert!(self.f.is_empty());
        assert_eq!(f.len(), self.w.len());
        self.message(bit, sum, |w| {
            if f.len() / 2 < WIDTH {
                scalar_measure(f.len(), bit, |j| G::from(BF128::from(f[j])), w)
            } else {
                measure(f.len(), bit, |j, b| base_pair(f, j, b), w)
            }
        })
    }
    pub fn bind(&mut self, bit: usize, alpha: BF128, next_bit: Option<usize>) {
        if let Some(f) = self.initial.take() {
            return self.bind_base(f, bit, alpha, next_bit);
        }
        let f = &self.f;
        self.measured = bind(
            f.len(),
            bit,
            G::from(alpha),
            next_bit,
            |j, b| pair(f, j, b),
            |j| f[j],
            &self.w,
            &mut self.f_scratch,
            &mut self.w_scratch,
        );
        core::mem::swap(&mut self.f, &mut self.f_scratch);
        core::mem::swap(&mut self.w, &mut self.w_scratch);
    }
    pub fn bind_base<B: Field>(
        &mut self,
        f: &[B],
        bit: usize,
        alpha: BF128,
        next_bit: Option<usize>,
    ) where
        BF128: ExtensionField<B>,
    {
        assert!(self.f.is_empty());
        self.measured = bind(
            f.len(),
            bit,
            G::from(alpha),
            next_bit,
            |j, b| base_pair(f, j, b),
            |j| G::from(BF128::from(f[j])),
            &self.w,
            &mut self.f_scratch,
            &mut self.w_scratch,
        );
        core::mem::swap(&mut self.f, &mut self.f_scratch);
        core::mem::swap(&mut self.w, &mut self.w_scratch);
    }
    pub fn table(&self) -> Vec<BF128> {
        assert!(self.initial.is_none());
        self.f.iter().copied().map(BF128::from).collect()
    }
}

fn scalar_measure(len: usize, bit: usize, f: impl Fn(usize) -> G, w: &[G]) -> [G; 2] {
    let mut sum = [G::ZERO; 2];
    for j in 0..len / 2 {
        let lo = source(j, bit);
        let hi = lo + (1 << bit);
        sum[0] += f(lo) * w[lo];
        sum[1] += (f(lo) + f(hi)) * (w[lo] + w[hi]);
    }
    sum
}

fn resize_output(out: &mut Vec<G>, len: usize) {
    if len >= PARALLEL && out.capacity() < len {
        // Rayon initializes disjoint output ranges, including their first touch.
        // Later rounds only truncate/reuse these buffers, with no clearing pass.
        *out = (0..len).into_par_iter().map(|_| G::ZERO).collect();
    } else {
        out.resize(len, G::ZERO);
    }
}

/// Bind f/w together and optionally measure the next round while their output
/// registers are hot. The cached message is observed only at its original
/// transcript position, after any intervening Merkle root.
#[allow(clippy::too_many_arguments)]
fn bind(
    len: usize,
    bit: usize,
    alpha: G,
    next: Option<usize>,
    f: impl Fn(usize, usize) -> (P, P) + Sync,
    scalar_f: impl Fn(usize) -> G + Sync,
    w: &[G],
    out_f: &mut Vec<G>,
    out_w: &mut Vec<G>,
) -> Option<(usize, [G; 2])> {
    assert_eq!(len, w.len());
    assert!(len.is_multiple_of(2 << bit));
    let half = len / 2;
    resize_output(out_f, half);
    resize_output(out_w, half);
    if let Some(next_bit) = next {
        assert!(half.is_multiple_of(2 << next_bit));
        let stride = 1 << next_bit;
        let group = |base: usize, of: &mut [G], ow: &mut [G]| {
            let (fl, fh) = of.split_at_mut(stride);
            let (wl, wh) = ow.split_at_mut(stride);
            let block = |offset: usize, fl: &mut [G], fh: &mut [G], wl: &mut [G], wh: &mut [G]| {
                let mut m = Measurement::new();
                let mut k = 0;
                while k + WIDTH <= fl.len() {
                    let j = base + offset + k;
                    let (a, b) = f(j, bit);
                    let (c, d) = f(j + stride, bit);
                    let (x, y) = pair(w, j, bit);
                    let (z, t) = pair(w, j + stride, bit);
                    let lo = a + (a + b) * alpha;
                    let hi = c + (c + d) * alpha;
                    let low = x + (x + y) * alpha;
                    let high = z + (z + t) * alpha;
                    *P::from_slice_mut(&mut fl[k..k + WIDTH]) = lo;
                    *P::from_slice_mut(&mut fh[k..k + WIDTH]) = hi;
                    *P::from_slice_mut(&mut wl[k..k + WIDTH]) = low;
                    *P::from_slice_mut(&mut wh[k..k + WIDTH]) = high;
                    m.push(lo, hi, low, high);
                    k += WIDTH;
                }
                let mut sum = m.finish();
                for k in k..fl.len() {
                    let j = source(base + offset + k, bit);
                    let h = source(base + offset + k + stride, bit);
                    let a = scalar_f(j);
                    let b = scalar_f(j + (1 << bit));
                    let c = scalar_f(h);
                    let d = scalar_f(h + (1 << bit));
                    fl[k] = a + alpha * (a + b);
                    fh[k] = c + alpha * (c + d);
                    wl[k] = w[j] + alpha * (w[j] + w[j + (1 << bit)]);
                    wh[k] = w[h] + alpha * (w[h] + w[h + (1 << bit)]);
                    sum[0] += fl[k] * wl[k];
                    sum[1] += (fl[k] + fh[k]) * (wl[k] + wh[k]);
                }
                sum
            };
            if half < PARALLEL || stride <= CHUNK {
                block(0, fl, fh, wl, wh)
            } else {
                fl.par_chunks_mut(CHUNK)
                    .zip(fh.par_chunks_mut(CHUNK))
                    .zip(wl.par_chunks_mut(CHUNK))
                    .zip(wh.par_chunks_mut(CHUNK))
                    .enumerate()
                    .par_fold_reduce(
                        || [G::ZERO; 2],
                        |sum, (chunk, (((fl, fh), wl), wh))| {
                            add(sum, block(chunk * CHUNK, fl, fh, wl, wh))
                        },
                        add,
                    )
            }
        };
        let sum = if half < PARALLEL {
            out_f
                .chunks_mut(2 * stride)
                .zip(out_w.chunks_mut(2 * stride))
                .enumerate()
                .fold([G::ZERO; 2], |sum, (i, (f, w))| {
                    add(sum, group(i * 2 * stride, f, w))
                })
        } else {
            out_f
                .par_chunks_mut(2 * stride)
                .zip(out_w.par_chunks_mut(2 * stride))
                .enumerate()
                .par_fold_reduce(
                    || [G::ZERO; 2],
                    |sum, (i, (f, w))| add(sum, group(i * 2 * stride, f, w)),
                    add,
                )
        };
        Some((next_bit, sum))
    } else {
        let block = |start: usize, of: &mut [G], ow: &mut [G]| {
            let mut k = 0;
            while k + WIDTH <= of.len() {
                let (a, b) = f(start + k, bit);
                let (x, y) = pair(w, start + k, bit);
                *P::from_slice_mut(&mut of[k..k + WIDTH]) = a + (a + b) * alpha;
                *P::from_slice_mut(&mut ow[k..k + WIDTH]) = x + (x + y) * alpha;
                k += WIDTH;
            }
            for k in k..of.len() {
                let j = source(start + k, bit);
                let a = scalar_f(j);
                of[k] = a + alpha * (a + scalar_f(j + (1 << bit)));
                ow[k] = w[j] + alpha * (w[j] + w[j + (1 << bit)]);
            }
        };
        if half < PARALLEL {
            block(0, out_f, out_w);
        } else {
            out_f
                .par_chunks_mut(CHUNK)
                .zip(out_w.par_chunks_mut(CHUNK))
                .enumerate()
                .for_each(|(i, (f, w))| block(i * CHUNK, f, w));
        }
        None
    }
}
