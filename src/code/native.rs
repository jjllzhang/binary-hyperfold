//! Native base-field storage; extension folding starts only after challenges.
use super::{
    base_code,
    folding::{dot_product, finish_rows, fold_into_layout, fold_into_layout_with_sink},
    matrix_h::*,
};
use crate::{
    BF16, BF32, BF64, BF128,
    parameters::{HyperFoldConfig, MatrixKind},
};
use p3_binary_dft::ButterflyField;
use p3_binary_field::{BinaryField4, TowerLevel};
use p3_commit::RowSink;
use p3_field::{Algebra, ExtensionField, PrimeCharacteristicRing, batch_multiplicative_inverse};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use rand::{Rng, RngExt};

/// The three native tower levels supported by the PCS (not a runtime backend abstraction).
pub trait NativeField: ButterflyField {
    const BITS: usize;
    fn from_bits(x: u128) -> Self;
    fn mul_f16(self, c: BinaryField4) -> Self;
    fn mul_omega(self) -> Self;
}
macro_rules! native_field {
    ($field:ty, $repr:ty, $bits:expr) => {
        impl NativeField for $field {
            const BITS: usize = $bits;
            #[inline]
            fn from_bits(x: u128) -> Self {
                Self::from_repr(x as $repr)
            }
            #[inline]
            fn mul_f16(self, c: BinaryField4) -> Self {
                const MASK: $repr = <$repr>::MAX / 15;
                let bits = self.to_repr();
                let mut out = 0 as $repr;
                for i in 0..4 {
                    let plane = (bits >> i) & MASK;
                    let image = (c * BinaryField4::from_repr(1 << i)).to_repr();
                    for j in 0..4 {
                        out ^=
                            (plane & (0 as $repr).wrapping_sub(((image >> j) & 1) as $repr)) << j;
                    }
                }
                Self::from_repr(out)
            }
            #[inline]
            fn mul_omega(self) -> Self {
                const MASK: $repr = <$repr>::MAX / 3;
                let bits = self.to_repr();
                let lo = bits & MASK;
                let hi = (bits >> 1) & MASK;
                Self::from_repr(hi | ((lo ^ hi) << 1))
            }
        }
    };
}
native_field!(BF16, u16, 16);
native_field!(BF32, u32, 32);
native_field!(BF64, u64, 64);
pub type Code64 = NativeCode<BF64>;
#[inline]
fn omega<B: NativeField>(x: B) -> B {
    x.mul_omega()
}
#[inline]
fn omega_plus_one<B: NativeField>(x: B) -> B {
    x + x.mul_omega()
}
#[inline]
fn zeta<B: NativeField>(x: B) -> B {
    x.mul_f16(BinaryField4::from_repr(12))
}
#[inline]
fn zeta_inv<B: NativeField>(x: B) -> B {
    x.mul_f16(BinaryField4::from_repr(11))
}
#[inline]
fn scalar<B: NativeField>(x: B, c: BinaryField4) -> B {
    x.mul_f16(c)
}
#[derive(Clone, Debug)]
pub struct NativeCode<B: NativeField> {
    pub r: usize,
    pub k: Vec<usize>,
    pub n: Vec<usize>,
    pub log_rate: usize,
    pub twists: Vec<Vec<B>>,
    pub inverse_twists: Vec<Vec<B>>,
}
pub fn apply_h128(x: &mut [BF128], inv: bool) {
    match (x.len(), inv) {
        (2, false) => apply_h2(x),
        (2, true) => apply_h2_inv(x),
        (4, false) => apply_h4(x),
        (4, true) => apply_h4_inv(x),
        (8, _) => apply_h8(x),
        _ => panic!("arity"),
    }
}
fn symbol<B: NativeField>(word: &RowMajorMatrix<B>, j: usize) -> B {
    if word.width == 1 {
        word.values[j]
    } else {
        let n = word.values.len() / word.width;
        word.values[(j % n) * word.width + j / n]
    }
}
impl<B: NativeField> NativeCode<B>
where
    BF128: ExtensionField<B>,
{
    pub fn setup(c: &HyperFoldConfig, rng: &mut impl Rng) -> Self {
        assert_eq!(c.params.base_field_bits, B::BITS);
        assert_eq!(c.params.matrix_kind, MatrixKind::Specialized);
        let mut twists = vec![];
        let mut inverse_twists = vec![];
        for i in 1..=c.d {
            let mut v = vec![B::ONE; c.n[i]];
            for row in v.chunks_mut(c.params.r) {
                for x in &mut row[1..] {
                    loop {
                        // Deliberately consume the exact u128 RNG stream of the baseline.
                        let bits = B::from_bits(rng.random::<u128>());
                        if bits != B::ZERO {
                            *x = bits;
                            break;
                        }
                    }
                }
            }
            inverse_twists.push(batch_multiplicative_inverse(&v));
            twists.push(v);
        }
        Self {
            r: c.params.r,
            k: c.k.clone(),
            n: c.n.clone(),
            log_rate: c.params.c.ilog2() as usize,
            twists,
            inverse_twists,
        }
    }
    pub fn encode(&self, layer: usize, message: &[B]) -> RowMajorMatrix<B> {
        self.encode_impl(layer, message, None)
    }

    /// Hand over only the requested layer's finished native-width rows.
    pub fn encode_with_sink(
        &self,
        layer: usize,
        message: &[B],
        sink: &RowSink<'_, B>,
    ) -> RowMajorMatrix<B> {
        self.encode_impl(layer, message, Some(sink))
    }

    fn encode_impl(
        &self,
        layer: usize,
        message: &[B],
        sink: Option<&RowSink<'_, B>>,
    ) -> RowMajorMatrix<B> {
        assert_eq!(message.len(), self.k[layer]);
        if layer == 0 {
            return if let Some(sink) = sink {
                base_code::encode_native_with_sink(message, self.log_rate, sink)
            } else {
                RowMajorMatrix::new(base_code::encode_native(message, self.log_rate), 1)
            };
        }
        let children: Vec<_> = message
            .par_chunks(self.k[layer - 1])
            .map(|m| self.encode(layer - 1, m))
            .collect();
        finish_rows(
            self.n[layer],
            self.r,
            B::ZERO,
            |j, row| {
                row[0] = symbol(&children[0], j);
                for a in 1..self.r {
                    row[a] = symbol(&children[a], j) * self.twists[layer - 1][j * self.r + a];
                }
                apply_h(row, false);
            },
            sink,
        )
    }
    pub fn fold_row_top(&self, layer: usize, j: usize, row: &[B], beta: &[BF128]) -> BF128 {
        let mut v = [B::ZERO; 8];
        v[..self.r].copy_from_slice(row);
        apply_h(&mut v[..self.r], true);
        // Invert H and unscale in the native field, including for BF16/BF32.
        // The first lane has inverse twist one and needs no multiplication.
        for (a, value) in v.iter_mut().enumerate().take(self.r).skip(1) {
            *value *= self.inverse_twists[layer - 1][j * self.r + a];
        }
        match self.r {
            2 => <BF128 as Algebra<B>>::mixed_dot_product::<2>(
                beta[..2].try_into().unwrap(),
                v[..2].try_into().unwrap(),
            ),
            4 => <BF128 as Algebra<B>>::mixed_dot_product::<4>(
                beta[..4].try_into().unwrap(),
                v[..4].try_into().unwrap(),
            ),
            8 => <BF128 as Algebra<B>>::mixed_dot_product::<8>(
                beta[..8].try_into().unwrap(),
                v[..8].try_into().unwrap(),
            ),
            _ => panic!("arity"),
        }
    }
    pub fn fold_row(&self, layer: usize, j: usize, row: &[BF128], beta: &[BF128]) -> BF128 {
        let mut v = [BF128::ZERO; 8];
        v[..self.r].copy_from_slice(row);
        apply_h128(&mut v[..self.r], true);
        for (a, value) in v.iter_mut().enumerate().take(self.r).skip(1) {
            *value *= self.inverse_twists[layer - 1][j * self.r + a];
        }
        dot_product(self.r, beta, &v)
    }
    /// Fold a native top oracle into BF128 and hand over destination rows once.
    pub fn fold_top_with_sink(
        &self,
        layer: usize,
        word: &RowMajorMatrix<B>,
        beta: &[BF128],
        sink: &RowSink<'_, BF128>,
    ) -> RowMajorMatrix<BF128> {
        fold_into_layout_with_sink(
            word.values.len() / self.r,
            if layer == 1 { 1 } else { self.r },
            |j| self.fold_row_top(layer, j, &word.values[j * self.r..(j + 1) * self.r], beta),
            Some(sink),
        )
    }

    /// Fold an intermediate BF128 oracle and hand over destination rows once.
    pub fn fold_with_sink(
        &self,
        layer: usize,
        word: &RowMajorMatrix<BF128>,
        beta: &[BF128],
        sink: &RowSink<'_, BF128>,
    ) -> RowMajorMatrix<BF128> {
        fold_into_layout_with_sink(
            word.values.len() / self.r,
            if layer == 1 { 1 } else { self.r },
            |j| self.fold_row(layer, j, &word.values[j * self.r..(j + 1) * self.r], beta),
            Some(sink),
        )
    }

    pub fn fold_top(
        &self,
        layer: usize,
        word: &RowMajorMatrix<B>,
        beta: &[BF128],
    ) -> RowMajorMatrix<BF128> {
        fold_into_layout(
            word.values.len() / self.r,
            if layer == 1 { 1 } else { self.r },
            |j| self.fold_row_top(layer, j, &word.values[j * self.r..(j + 1) * self.r], beta),
        )
    }
    pub fn fold(
        &self,
        layer: usize,
        word: &RowMajorMatrix<BF128>,
        beta: &[BF128],
    ) -> RowMajorMatrix<BF128> {
        fold_into_layout(
            word.values.len() / self.r,
            if layer == 1 { 1 } else { self.r },
            |j| self.fold_row(layer, j, &word.values[j * self.r..(j + 1) * self.r], beta),
        )
    }
}

fn apply_h2_base<B: NativeField>(x: &mut [B]) {
    let a = x[0] + x[1];
    x[1] = a + omega_plus_one(x[1]);
    x[0] = a;
}
fn apply_h2_inv_base<B: NativeField>(x: &mut [B]) {
    x[1] = omega(x[0] + x[1]);
    x[0] += x[1];
}
fn apply_h4_base<B: NativeField>(x: &mut [B]) {
    let [mut a, mut b, mut c, mut d]: [B; 4] = x.try_into().unwrap();
    c += d;
    a += b;
    b = zeta(b + c);
    c += zeta(a);
    d = zeta(d + b);
    a += d;
    b += a;
    d += c;
    x.copy_from_slice(&[a, b, c, d]);
}
fn apply_h4_inv_base<B: NativeField>(x: &mut [B]) {
    let [mut a, mut b, mut c, mut d]: [B; 4] = x.try_into().unwrap();
    d += c;
    b += a;
    a += d;
    d = zeta_inv(d) + b;
    c += zeta(a);
    b = zeta_inv(b) + c;
    a += b;
    c += d;
    x.copy_from_slice(&[a, b, c, d]);
}
fn apply_h8_base<B: NativeField>(x: &mut [B]) {
    let v0 = x[0] + x[1];
    let v1 = x[2] + x[3];
    let v2 = x[2] + omega(v1);
    let v3 = x[0] + v2;
    let v4 = v0 + v1;
    let v5 = x[4] + x[5];
    let v6 = x[4] + scalar(v5, BinaryField4::from_repr(12));
    let v7 = x[6] + x[7];
    let v8 = x[6] + scalar(v7, BinaryField4::from_repr(14));
    let v9 = v6 + v8;
    let v10 = v5 + v7;
    let v11 = v6 + omega(v9);
    let v12 = v5 + omega(v10);
    let v13 = x[0] + v11;
    let v14 = v0 + v12;
    let v15 = v3 + v9;
    let v16 = v4 + v10;
    let v17 = x[0] + omega(v13);
    let v18 = v0 + omega(v14);
    let v19 = v3 + omega(v15);
    let v20 = v4 + omega(v16);
    let v21 = v17 + v13;
    let v22 = v18 + v14;
    let v23 = v19 + v15;
    let v24 = v20 + v16;
    let v25 = v17 + scalar(v19, BinaryField4::from_repr(12));
    let v26 = v18 + scalar(v20, BinaryField4::from_repr(12));
    let v27 = v25 + v19;
    let v28 = v26 + v20;
    let v29 = v25 + scalar(v26, BinaryField4::from_repr(6));
    let v30 = v29 + v26;
    let v31 = v27 + scalar(v28, BinaryField4::from_repr(4));
    let v32 = v31 + v28;
    let v33 = v21 + scalar(v23, BinaryField4::from_repr(14));
    let v34 = v22 + scalar(v24, BinaryField4::from_repr(14));
    let v35 = v33 + v23;
    let v36 = v34 + v24;
    let v37 = v33 + scalar(v34, BinaryField4::from_repr(10));
    let v38 = v37 + v34;
    let v39 = v35 + scalar(v36, BinaryField4::from_repr(8));
    let v40 = v39 + v36;
    x.copy_from_slice(&[v29, v30, v31, v32, v37, v38, v39, v40]);
}
fn apply_h8_inv_base<B: NativeField>(x: &mut [B]) {
    apply_h8_base(x)
}

pub fn apply_h<B: NativeField>(x: &mut [B], inv: bool) {
    match (x.len(), inv) {
        (2, false) => apply_h2_base(x),
        (2, true) => apply_h2_inv_base(x),
        (4, false) => apply_h4_base(x),
        (4, true) => apply_h4_inv_base(x),
        (8, false) => apply_h8_base(x),
        (8, true) => apply_h8_inv_base(x),
        _ => panic!("arity"),
    }
}
pub fn apply_h64(x: &mut [BF64], inv: bool) {
    apply_h(x, inv);
}
