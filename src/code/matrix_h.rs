use super::subfield::*;
pub use crate::parameters::MatrixKind;
use crate::{BF64, BF128};
use p3_binary_field::{BinaryField4, TowerLevel};
use p3_field::{Field, PrimeCharacteristicRing};

pub fn apply_h2(x: &mut [BF128]) {
    let a = x[0] + x[1];
    x[1] = a + mul_omega_plus_1(x[1]);
    x[0] = a;
}
pub fn apply_h2_inv(x: &mut [BF128]) {
    x[1] = mul_omega(x[0] + x[1]);
    x[0] += x[1];
}
pub fn apply_h4(x: &mut [BF128]) {
    let [mut a, mut b, mut c, mut d]: [BF128; 4] = x.try_into().unwrap();
    c += d;
    a += b;
    b = mul_zeta(b + c);
    c += mul_zeta(a);
    d = mul_zeta(d + b);
    a += d;
    b += a;
    d += c;
    x.copy_from_slice(&[a, b, c, d]);
}
pub fn apply_h4_inv(x: &mut [BF128]) {
    let [mut a, mut b, mut c, mut d]: [BF128; 4] = x.try_into().unwrap();
    d += c;
    b += a;
    a += d;
    d = mul_zeta_inv(d) + b;
    c += mul_zeta(a);
    b = mul_zeta_inv(b) + c;
    a += b;
    c += d;
    x.copy_from_slice(&[a, b, c, d]);
}
pub fn apply_h8(x: &mut [BF128]) {
    let v0 = x[0] + x[1];
    let v1 = x[2] + x[3];
    let v2 = x[2] + mul_omega(v1);
    let v3 = x[0] + v2;
    let v4 = v0 + v1;
    let v5 = x[4] + x[5];
    let v6 = x[4] + mul_f16_const(v5, BinaryField4::from_repr(12));
    let v7 = x[6] + x[7];
    let v8 = x[6] + mul_f16_const(v7, BinaryField4::from_repr(14));
    let v9 = v6 + v8;
    let v10 = v5 + v7;
    let v11 = v6 + mul_omega(v9);
    let v12 = v5 + mul_omega(v10);
    let v13 = x[0] + v11;
    let v14 = v0 + v12;
    let v15 = v3 + v9;
    let v16 = v4 + v10;
    let v17 = x[0] + mul_omega(v13);
    let v18 = v0 + mul_omega(v14);
    let v19 = v3 + mul_omega(v15);
    let v20 = v4 + mul_omega(v16);
    let v21 = v17 + v13;
    let v22 = v18 + v14;
    let v23 = v19 + v15;
    let v24 = v20 + v16;
    let v25 = v17 + mul_f16_const(v19, BinaryField4::from_repr(12));
    let v26 = v18 + mul_f16_const(v20, BinaryField4::from_repr(12));
    let v27 = v25 + v19;
    let v28 = v26 + v20;
    let v29 = v25 + mul_f16_const(v26, BinaryField4::from_repr(6));
    let v30 = v29 + v26;
    let v31 = v27 + mul_f16_const(v28, BinaryField4::from_repr(4));
    let v32 = v31 + v28;
    let v33 = v21 + mul_f16_const(v23, BinaryField4::from_repr(14));
    let v34 = v22 + mul_f16_const(v24, BinaryField4::from_repr(14));
    let v35 = v33 + v23;
    let v36 = v34 + v24;
    let v37 = v33 + mul_f16_const(v34, BinaryField4::from_repr(10));
    let v38 = v37 + v34;
    let v39 = v35 + mul_f16_const(v36, BinaryField4::from_repr(8));
    let v40 = v39 + v36;
    x.copy_from_slice(&[v29, v30, v31, v32, v37, v38, v39, v40]);
}
pub fn apply_h8_inv(x: &mut [BF128]) {
    apply_h8(x)
}

pub fn apply_h2_64(x: &mut [BF64]) {
    super::native::apply_h(x, false);
}
pub fn apply_h2_inv_64(x: &mut [BF64]) {
    super::native::apply_h(x, true);
}
pub fn apply_h4_64(x: &mut [BF64]) {
    super::native::apply_h(x, false);
}
pub fn apply_h4_inv_64(x: &mut [BF64]) {
    super::native::apply_h(x, true);
}
pub fn apply_h8_64(x: &mut [BF64]) {
    super::native::apply_h(x, false);
}
pub fn apply_h8_inv_64(x: &mut [BF64]) {
    super::native::apply_h(x, true);
}

#[derive(Clone, Debug)]
pub struct LocalMatrix {
    pub r: usize,
    pub kind: MatrixKind,
    pub dense: Vec<BF128>,
    inverse: Vec<BF128>,
}
impl LocalMatrix {
    pub fn new(r: usize, kind: MatrixKind) -> Self {
        let mut dense = vec![BF128::ZERO; r * r];
        if kind == MatrixKind::Specialized {
            for j in 0..r {
                let mut e = vec![BF128::ZERO; r];
                e[j] = BF128::ONE;
                apply_specialized(&mut e, false);
                for i in 0..r {
                    dense[i * r + j] = e[i];
                }
            }
        } else {
            for i in 0..r {
                for j in 0..r {
                    dense[i * r + j] = BF128::from_repr((i ^ (r + j)) as u128).inverse();
                }
            }
        }
        let inverse = invert(&dense, r);
        Self {
            r,
            kind,
            dense,
            inverse,
        }
    }
    pub fn apply(&self, x: &mut [BF128], inverse: bool) {
        if self.kind == MatrixKind::Specialized {
            apply_specialized(x, inverse)
        } else {
            let a = if inverse { &self.inverse } else { &self.dense };
            let mut y = [BF128::ZERO; 8];
            for i in 0..self.r {
                for j in 0..self.r {
                    y[i] += a[i * self.r + j] * x[j];
                }
            }
            x.copy_from_slice(&y[..self.r]);
        }
    }
}
fn apply_specialized(x: &mut [BF128], inv: bool) {
    match (x.len(), inv) {
        (2, false) => apply_h2(x),
        (2, true) => apply_h2_inv(x),
        (4, false) => apply_h4(x),
        (4, true) => apply_h4_inv(x),
        (8, _) => apply_h8(x),
        _ => panic!("arity"),
    }
}
pub fn invert(a: &[BF128], r: usize) -> Vec<BF128> {
    let mut a = a.to_vec();
    let mut out = vec![BF128::ZERO; r * r];
    for i in 0..r {
        out[i * r + i] = BF128::ONE;
    }
    for col in 0..r {
        let pivot = (col..r).find(|&i| a[i * r + col] != BF128::ZERO).unwrap();
        for j in 0..r {
            a.swap(col * r + j, pivot * r + j);
            out.swap(col * r + j, pivot * r + j);
        }
        let inv = a[col * r + col].inverse();
        for j in 0..r {
            a[col * r + j] *= inv;
            out[col * r + j] *= inv;
        }
        for i in 0..r {
            if i == col {
                continue;
            }
            let m = a[i * r + col];
            for j in 0..r {
                let ac = a[col * r + j];
                let oc = out[col * r + j];
                a[i * r + j] += m * ac;
                out[i * r + j] += m * oc;
            }
        }
    }
    out
}
