use crate::BF128;
use p3_binary_field::{BinaryChallenger, TowerLevel};
use p3_blake3::Blake3;
use p3_challenger::{CanObserve, CanSample, CanSampleBits, HashChallenger};
type Challenger = BinaryChallenger<BF128, HashChallenger<u8, Blake3, 32>>;
#[derive(Clone, Debug)]
pub struct ProverTranscript {
    inner: Challenger,
}
pub type VerifierTranscript = ProverTranscript;
impl ProverTranscript {
    pub fn new(context: &[u8]) -> Self {
        Self {
            inner: Challenger::from_hasher(context.to_vec(), Blake3),
        }
    }
    pub fn observe(&mut self, value: BF128) {
        self.inner.observe(value)
    }
    pub fn observe_fields(&mut self, v: &[BF128]) {
        for &x in v {
            self.observe(x)
        }
    }
    pub fn observe_bytes(&mut self, v: &[u8]) {
        self.observe(BF128::from_repr(v.len() as u128));
        for chunk in v.chunks(16) {
            let mut b = [0; 16];
            b[..chunk.len()].copy_from_slice(chunk);
            self.observe(BF128::from_repr(u128::from_le_bytes(b)));
        }
    }
    pub fn sample(&mut self) -> BF128 {
        self.inner.sample()
    }
    pub fn point(&mut self, n: usize) -> Vec<BF128> {
        (0..n).map(|_| self.sample()).collect()
    }
    pub fn index(&mut self, n: usize) -> usize {
        assert!(n.is_power_of_two());
        self.inner.sample_bits(n.ilog2() as usize)
    }
}
