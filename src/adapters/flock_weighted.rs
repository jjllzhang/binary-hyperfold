//! Flock's ring-switched transparent inner products, opened directly by the
//! interleaved HyperFold sumcheck plus a short bottom-table sumcheck.
use super::*;
use crate::{
    pcs::weighted::{TransparentWeight, WeightedClaim, WeightedHyperFoldProof},
    transcript::ProverTranscript,
};
use p3_field::{Field, PackedValue};
use p3_maybe_rayon::prelude::*;
use serde::{Deserialize, Serialize};

// The same 16-byte lookup as Flock's internal deferred fold, using safe
// indexing. Each table is built once per claim, not once per chunk.
#[inline]
fn fold_one_slot(element: F128, table: &[F128]) -> F128 {
    debug_assert_eq!(table.len(), 16 * 256);
    let bytes = (element.lo as u128 | ((element.hi as u128) << 64)).to_le_bytes();
    let mut sums: [F128; 8] = std::array::from_fn(|i| {
        table[(2 * i) * 256 + bytes[2 * i] as usize]
            + table[(2 * i + 1) * 256 + bytes[2 * i + 1] as usize]
    });
    for stride in [4, 2, 1] {
        for i in 0..stride {
            sums[i] += sums[i + stride];
        }
    }
    sums[0]
}

// The Flock batched prover retains the two equality factors; the 128-bit
// ring-switch fold is reconstructed only for slots actually consumed by PCS.
struct RingSwitchWeight {
    table: Option<Vec<F128>>,
    factors: Option<(Vec<F128>, Vec<F128>)>,
    z_vals: Vec<F128>,
    eq_r_dprime: Vec<F128>,
}
impl TransparentWeight for RingSwitchWeight {
    fn table(&self) -> Option<&[BF128]> {
        None
    }
    fn add_scaled_into(&self, out: &mut [Ghash128], scale: BF128) {
        if let Some(src) = &self.table {
            assert_eq!(out.len(), src.len());
            type P = <Ghash128 as Field>::Packing;
            let scale = Ghash128::from(scale);
            let factor = P::broadcast(scale);
            out.par_chunks_mut(4096)
                .zip(src.par_chunks(4096))
                .for_each(|(dst, src)| {
                    let mut j = 0;
                    while j + P::WIDTH <= src.len() {
                        let x = P::from_fn(|lane| {
                            let v = src[j + lane];
                            Ghash128::from_repr(v.lo as u128 | ((v.hi as u128) << 64))
                        });
                        *P::from_slice_mut(&mut dst[j..j + P::WIDTH]) += x * factor;
                        j += P::WIDTH;
                    }
                    for (out, &v) in dst[j..].iter_mut().zip(&src[j..]) {
                        *out += Ghash128::from_repr(v.lo as u128 | ((v.hi as u128) << 64)) * scale;
                    }
                });
            return;
        }
        let (lo, hi) = self
            .factors
            .as_ref()
            .expect("prover needs ring-switch factors");
        assert_eq!(out.len(), lo.len() * hi.len());
        let table = ring_switch::build_fold_byte_table(&self.eq_r_dprime);
        let scale = Ghash128::from(scale);
        type P = <Ghash128 as Field>::Packing;
        let factor = P::broadcast(scale);
        out.par_chunks_mut(4096)
            .enumerate()
            .for_each(|(chunk, dst)| {
                let start = chunk * 4096;
                let value_at = |index: usize| {
                    let element = lo[index % lo.len()] * hi[index / lo.len()];
                    let value = fold_one_slot(element, &table);
                    // Flock F128 and Ghash128 use the same polynomial basis.
                    Ghash128::from_repr(value.lo as u128 | ((value.hi as u128) << 64))
                };
                let mut j = 0;
                while j + P::WIDTH <= dst.len() {
                    let x = P::from_fn(|lane| value_at(start + j + lane));
                    *P::from_slice_mut(&mut dst[j..j + P::WIDTH]) += x * factor;
                    j += P::WIDTH;
                }
                for (offset, value) in dst[j..].iter_mut().enumerate() {
                    *value += value_at(start + j + offset) * scale;
                }
            });
    }
    fn evaluate(&self, point: &[BF128]) -> BF128 {
        assert_eq!(point.len(), self.z_vals.len());
        let query: Vec<_> = point.iter().copied().map(to_flock).collect();
        from_flock(ring_switch::eval_rs_eq(
            &self.z_vals,
            &query,
            &self.eq_r_dprime,
        ))
    }
    fn observe(&self, tr: &mut ProverTranscript) {
        tr.observe_bytes(b"flock-ring-switch-weight-v1");
        tr.observe_bytes(&(self.z_vals.len() as u64).to_le_bytes());
        for &x in &self.z_vals {
            tr.observe(from_flock(x));
        }
        for &x in &self.eq_r_dprime {
            tr.observe(from_flock(x));
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlockHyperFoldProof {
    pub commitment: Commitment,
    pub zerocheck: ZerocheckProof,
    pub lincheck: LincheckProof,
    pub ring_switch: Vec<ring_switch::RingSwitchProof>,
    pub opening: WeightedHyperFoldProof,
}
impl FlockHyperFoldProof {
    pub fn wire_size_in_bytes(&self) -> usize {
        bincode::serialized_size(self).unwrap() as usize
    }
}

/// Prove Flock's two ring-switched claims with the weighted HyperFold PCS.
pub fn prove_r1cs_with_binary_hyperfold(
    r: &BlockR1cs,
    pcs: &BinaryHyperFoldPcs,
    z: Vec<F128>,
    a: Vec<F128>,
    b: Vec<F128>,
    z_lincheck: &[u8],
    circuit: &dyn LincheckCircuit,
) -> (FlockHyperFoldProof, PhaseTimings) {
    assert!(r.c0_is_identity());
    assert_eq!(r.m - 7, pcs.config.params.mu);
    assert_eq!(z.len(), 1usize << (r.m - 7));
    let mut timing = PhaseTimings::default();
    let t = Instant::now();
    let tower = z.par_iter().copied().map(from_flock).collect();
    let (cm, data) = pcs.commit(tower);
    timing.commit_ms = elapsed(t);
    let mut ch = FsChallenger::new(b"binary-hyperfold-flock");
    bind(&mut ch, r, pcs, &cm);
    let aa = packed_bytes(&a);
    let bb = packed_bytes(&b);
    let zz = packed_bytes(&z);
    let t = Instant::now();
    let (zp, zc) = zerocheck::prove_packed_padded(aa, bb, zz, r.m, &r.padding_spec(), &mut ch);
    timing.zerocheck_ms = elapsed(t);
    let x = r.x_ab_from_multilinear_values(SkipPoint::Phi8(zc.z), &zc.mlv_challenges);
    let t = Instant::now();
    let (lp, lc, _) = lincheck::prove_padded_capture_z_vec(
        z_lincheck,
        r.m,
        r.k_log,
        r.k_skip,
        r.useful_bits,
        circuit,
        &x,
        &mut ch,
    );
    timing.lincheck_ms = elapsed(t);
    let claims = claims(r, &zc, &lc, &x);
    let t = Instant::now();
    let mut proofs = Vec::with_capacity(2);
    let mut weights = Vec::with_capacity(2);
    let mut values = Vec::with_capacity(2);
    if z.len() >= 1 << 21 && current_num_threads() >= 8 {
        let fulls: Vec<_> = claims.iter().map(|claim| outer(&claim.point)).collect();
        let refs: Vec<_> = fulls.iter().map(Vec::as_slice).collect();
        // The batched prover defers the dense fold, but samples an extra Flock
        // batch coefficient after both proofs. Use a snapshot, then replay
        // only the original two-proof transcript on `ch`. PCS gamma is sampled
        // later, after observing both weighted claims.
        let mut snapshot = ch.clone();
        let (batch, _) = ring_switch::prove_batched(&z, &refs, &mut snapshot);
        for ((claim, full), (proof, out)) in claims.iter().zip(&fulls).zip(batch) {
            let descriptor = ring_switch::verify_succinct_with_grinding(
                claim.value,
                &claim.point.z_skip.weights(r.k_skip),
                full,
                &proof,
                0,
                &mut ch,
            )
            .expect("ring-switch prover returned an invalid claim");
            debug_assert_eq!(out.sumcheck_claim, descriptor.sumcheck_claim);
            let factors = match out.rs_eq_ind {
                ring_switch::RsEqInd::DeferredDense { eq_lo, eq_hi, .. } => (eq_lo, eq_hi),
                _ => panic!("ring-switch batch must return deferred dense factors"),
            };
            weights.push(RingSwitchWeight {
                table: None,
                factors: Some(factors),
                z_vals: full[1..].to_vec(),
                eq_r_dprime: descriptor.eq_r_dprime,
            });
            values.push(from_flock(descriptor.sumcheck_claim));
            proofs.push(proof);
        }
    } else {
        // For small instances or few workers, the materialized vector plus
        // the packed GHASH combine wins over computing every slot on demand.
        for claim in &claims {
            let full = outer(&claim.point);
            let mut snapshot = ch.clone();
            let (proof, out) = ring_switch::prove(&z, &full, &mut ch);
            snapshot.observe_label(b"flock-ring-switch-v0");
            snapshot.observe_f128_slice(&proof.s_hat_v);
            let r_dprime = snapshot.sample_f128_vec(7);
            let eq_r_dprime = ring_switch::build_eq_split(&r_dprime, 7).0;
            weights.push(RingSwitchWeight {
                table: Some(out.rs_eq_ind),
                factors: None,
                z_vals: full[1..].to_vec(),
                eq_r_dprime,
            });
            values.push(from_flock(out.sumcheck_claim));
            proofs.push(proof);
        }
    }
    timing.ring_switch_ms = elapsed(t);
    let inputs: Vec<_> = weights
        .iter()
        .zip(&values)
        .map(|(weight, &value)| WeightedClaim { weight, value })
        .collect();
    let t = Instant::now();
    let opening = pcs.open_weighted(&data, &inputs);
    timing.open_ms = elapsed(t);
    (
        FlockHyperFoldProof {
            commitment: cm,
            zerocheck: zp,
            lincheck: lp,
            ring_switch: proofs,
            opening,
        },
        timing,
    )
}

pub fn verify_r1cs_with_binary_hyperfold(
    r: &BlockR1cs,
    pcs: &BinaryHyperFoldPcs,
    p: &FlockHyperFoldProof,
    circuit: &dyn LincheckCircuit,
) -> bool {
    if !r.c0_is_identity() || r.m < 7 || r.m - 7 != pcs.config.params.mu || p.ring_switch.len() != 2
    {
        return false;
    }
    let mut ch = FsChallenger::new(b"binary-hyperfold-flock");
    bind(&mut ch, r, pcs, &p.commitment);
    let Ok(zc) = zerocheck::verify(r.m, &p.zerocheck, &mut ch) else {
        return false;
    };
    let x = r.x_ab_from_multilinear_values(SkipPoint::Phi8(zc.z), &zc.mlv_challenges);
    let Ok(lc) = lincheck::verify(
        r.m,
        r.k_log,
        r.k_skip,
        circuit,
        &x,
        zc.a_eval,
        zc.b_eval,
        &p.lincheck,
        &mut ch,
    ) else {
        return false;
    };
    let claims = claims(r, &zc, &lc, &x);
    let mut weights = Vec::with_capacity(2);
    let mut values = Vec::with_capacity(2);
    for (claim, rp) in claims.iter().zip(&p.ring_switch) {
        let full = outer(&claim.point);
        if rp.s_hat_v.len() != 128 {
            return false;
        }
        let Ok(out) = ring_switch::verify_succinct_with_grinding(
            claim.value,
            &claim.point.z_skip.weights(r.k_skip),
            &full,
            rp,
            0,
            &mut ch,
        ) else {
            return false;
        };
        values.push(from_flock(out.sumcheck_claim));
        weights.push(RingSwitchWeight {
            table: None,
            factors: None,
            z_vals: full[1..].to_vec(),
            eq_r_dprime: out.eq_r_dprime,
        });
    }
    let inputs: Vec<_> = weights
        .iter()
        .zip(&values)
        .map(|(weight, &value)| WeightedClaim { weight, value })
        .collect();
    pcs.verify_weighted(&p.commitment, &inputs, &p.opening)
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    #[test]
    fn ring_switch_weight_evaluation_matches_dense_table() {
        use rand::{RngExt, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(411);
        let z: Vec<F128> = (0..16)
            .map(|_| F128::new(rng.random(), rng.random()))
            .collect();
        let point: Vec<F128> = (0..5)
            .map(|_| F128::new(rng.random(), rng.random()))
            .collect();
        let mut ch = FsChallenger::new(b"weight-eval-test");
        let mut snapshot = ch.clone();
        let (proof, output) = ring_switch::prove(&z, &point, &mut ch);
        snapshot.observe_label(b"flock-ring-switch-v0");
        snapshot.observe_f128_slice(&proof.s_hat_v);
        let eq_r_dprime = ring_switch::build_eq_split(&snapshot.sample_f128_vec(7), 7).0;
        let factors =
            ring_switch::build_eq_split(&point[1..], ring_switch::split_n_lo(point.len() - 1));
        let weight = RingSwitchWeight {
            table: None,
            factors: Some(factors),
            z_vals: point[1..].to_vec(),
            eq_r_dprime,
        };
        let dense: Vec<_> = output.rs_eq_ind.into_iter().map(from_flock).collect();
        let scale = from_flock(F128::new(rng.random(), rng.random()));
        let mut combined = vec![Ghash128::ZERO; dense.len()];
        weight.add_scaled_into(&mut combined, scale);
        for (&actual, &value) in combined.iter().zip(&dense) {
            assert_eq!(actual, Ghash128::from(value * scale));
        }
        for index in 0..z.len() {
            let query: Vec<_> = (0..4)
                .map(|bit| {
                    if index & (1 << bit) == 0 {
                        BF128::ZERO
                    } else {
                        BF128::ONE
                    }
                })
                .collect();
            assert_eq!(weight.evaluate(&query), dense[index]);
        }
        for _ in 0..8 {
            let query: Vec<BF128> = (0..4)
                .map(|_| from_flock(F128::new(rng.random(), rng.random())))
                .collect();
            assert_eq!(
                weight.evaluate(&query),
                crate::sumcheck::evaluate(&dense, &query)
            );
        }
    }
}
