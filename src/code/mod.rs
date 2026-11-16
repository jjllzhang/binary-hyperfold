pub mod base_code;
pub mod folding;
pub mod matrix_h;
pub mod subfield;
use crate::{BF128, parameters::HyperFoldConfig};
use matrix_h::LocalMatrix;
use p3_binary_field::TowerLevel;
use p3_field::{PrimeCharacteristicRing, batch_multiplicative_inverse};
use rand::{Rng, RngExt};

#[derive(Clone, Debug)]
pub struct HyperFoldableCode {
    pub r: usize,
    pub k: Vec<usize>,
    pub n: Vec<usize>,
    pub log_rate: usize,
    pub matrix: LocalMatrix,
    pub twists: Vec<Vec<BF128>>,
    pub inverse_twists: Vec<Vec<BF128>>,
}
impl HyperFoldableCode {
    pub fn setup(config: &HyperFoldConfig, rng: &mut impl Rng) -> Self {
        let r = config.params.r;
        let w = config.params.base_field_bits;
        let mask = if w == 128 {
            u128::MAX
        } else {
            (1u128 << w) - 1
        };
        let mut twists = vec![];
        let mut inverse_twists = vec![];
        for i in 1..=config.d {
            let mut v = vec![BF128::ONE; config.n[i]];
            for row in v.chunks_mut(r) {
                for x in &mut row[1..] {
                    loop {
                        let bits = rng.random::<u128>() & mask;
                        if bits != 0 {
                            *x = BF128::from_repr(bits);
                            break;
                        }
                    }
                }
            }
            inverse_twists.push(batch_multiplicative_inverse(&v));
            twists.push(v);
        }
        Self {
            r,
            k: config.k.clone(),
            n: config.n.clone(),
            log_rate: config.params.c.ilog2() as usize,
            matrix: LocalMatrix::new(r, config.params.matrix_kind),
            twists,
            inverse_twists,
        }
    }
}

pub mod native64;
