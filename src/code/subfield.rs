//! The specified GF(16) embedded in Plonky3's tower (not raw polynomial-basis bits).
use crate::{BF64, BF128};
use p3_binary_field::{BinaryField2, BinaryField4, TowerLevel};

/// Images of the four tower GF(16) basis vectors under each scalar.
/// Fixed at compile time; tested against the pinned field implementation.
#[rustfmt::skip]
pub const F16_IMAGES: [[u8;4];16] = [
    [0, 0, 0, 0],
    [1, 2, 4, 8],
    [2, 3, 8, 12],
    [3, 1, 12, 4],
    [4, 8, 9, 14],
    [5, 10, 13, 6],
    [6, 11, 1, 2],
    [7, 9, 5, 10],
    [8, 12, 14, 7],
    [9, 14, 10, 15],
    [10, 15, 6, 11],
    [11, 13, 2, 3],
    [12, 4, 7, 9],
    [13, 6, 3, 1],
    [14, 7, 15, 5],
    [15, 5, 11, 13],
];

const fn zeta_bits() -> u8 {
    12
}
pub fn f16(x: u8) -> BF128 {
    BF128::from_repr(x as u128)
}
pub fn zeta() -> BF128 {
    f16(zeta_bits())
}
pub fn omega() -> BF128 {
    f16(2)
}
pub fn tau() -> BF128 {
    let z = zeta();
    z * z * z + z
}
#[inline]
pub fn mul_omega(x: BF128) -> BF128 {
    x * BinaryField2::from_repr(2)
}
#[inline]
pub fn mul_omega_plus_1(x: BF128) -> BF128 {
    x + mul_omega(x)
}

/// Four bit planes, each comprising one coordinate from every tower GF(16) chunk.
/// Multiplication by a GF(16) scalar is a 4x4 GF(2) linear map on all 32 chunks.
#[inline]
pub fn mul_f16_const(x: BF128, c: BinaryField4) -> BF128 {
    const MASK: u128 = 0x11111111111111111111111111111111;
    let bits = x.to_repr();
    let mut out = 0u128;
    for i in 0..4 {
        let plane = (bits >> i) & MASK;
        let image = F16_IMAGES[c.to_repr() as usize][i];
        for j in 0..4 {
            out ^= (plane & 0u128.wrapping_sub(((image >> j) & 1) as u128)) << j;
        }
    }
    BF128::from_repr(out)
}
#[inline]
pub fn mul_zeta(x: BF128) -> BF128 {
    mul_f16_const(x, BinaryField4::from_repr(zeta_bits()))
}
#[inline]
pub fn mul_zeta_inv(x: BF128) -> BF128 {
    mul_f16_const(x, BinaryField4::from_repr(11))
}

/// Native tower GF(2^64) nibble map; the scalar itself is a GF(16) value.
#[inline]
pub fn mul_f16_const_64(x: BF64, c: BinaryField4) -> BF64 {
    super::native::NativeField::mul_f16(x, c)
}
#[inline]
pub fn mul_omega_64(x: BF64) -> BF64 {
    super::native::NativeField::mul_omega(x)
}
#[inline]
pub fn mul_omega_plus_1_64(x: BF64) -> BF64 {
    x + mul_omega_64(x)
}
#[inline]
pub fn mul_zeta_64(x: BF64) -> BF64 {
    mul_f16_const_64(x, BinaryField4::from_repr(zeta_bits()))
}
#[inline]
pub fn mul_zeta_inv_64(x: BF64) -> BF64 {
    mul_f16_const_64(x, BinaryField4::from_repr(11))
}

#[cfg(test)]
mod constant_images {
    use super::*;
    #[test]
    fn every_image_matches_field_basis_product() {
        for c in 0..16 {
            for i in 0..4 {
                assert_eq!(
                    F16_IMAGES[c][i],
                    (BinaryField4::from_repr(c as u8) * BinaryField4::from_repr(1 << i)).to_repr()
                );
            }
        }
    }
}
