//! Option B: Flock Boolean PIOP -> independent ring-switch sumcheck -> HyperFold.
//! The extra linear pass is intentionally not conflated with Ligerito's fused pass.
use crate::{
    BF128,
    pcs::{BinaryHyperFoldPcs, Commitment, proof::HyperFoldProof},
    sumcheck::{
        SumcheckPolynomial,
        prover::{fold_table, polynomial},
    },
};
use flock_core::{
    lincheck::{self, LincheckCircuit, LincheckProof, SkipPoint},
    pcs::ring_switch,
    proof::ZClaim,
    r1cs::BlockR1cs,
    zerocheck::{self, ZerocheckProof},
};
use flock_field::F128;
use flock_transcript::challenger::{Challenger, FsChallenger};
use p3_binary_field::{Ghash128, TowerLevel};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};
use std::time::Instant;

pub fn from_flock(x: F128) -> BF128 {
    Ghash128::from_repr(x.lo as u128 | ((x.hi as u128) << 64)).into()
}
pub fn to_flock(x: BF128) -> F128 {
    let v = Ghash128::from(x).to_repr();
    F128::new(v as u64, (v >> 64) as u64)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlockHyperFoldProof {
    pub commitment: Commitment,
    pub zerocheck: ZerocheckProof,
    pub lincheck: LincheckProof,
    pub ring_switch: Vec<ring_switch::RingSwitchProof>,
    pub reduction: Vec<SumcheckPolynomial>,
    pub packed_evaluation: BF128,
    pub opening: HyperFoldProof,
}
impl FlockHyperFoldProof {
    pub fn wire_size_in_bytes(&self) -> usize {
        bincode::serialized_size(self).unwrap() as usize
    }
}
#[derive(Clone, Debug, Default)]
pub struct PhaseTimings {
    pub commit_ms: f64,
    pub zerocheck_ms: f64,
    pub lincheck_ms: f64,
    pub ring_switch_ms: f64,
    pub open_ms: f64,
}
fn elapsed(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}
fn bind(ch: &mut impl Challenger, r: &BlockR1cs, pcs: &BinaryHyperFoldPcs, cm: &Commitment) {
    ch.observe_label(b"flock-binary-hyperfold-option-b-v1");
    ch.observe_bytes(&r.statement_digest());
    ch.observe_bytes(&bincode::serialize(&pcs.config).unwrap());
    ch.observe_bytes(&pcs.seed);
    ch.observe_bytes(&bincode::serialize(cm).unwrap());
}
fn outer(p: &lincheck::QuirkyPoint) -> Vec<F128> {
    p.x_inner_rest.iter().chain(&p.x_outer).copied().collect()
}
fn claims(
    r: &BlockR1cs,
    zc: &zerocheck::ZerocheckClaim,
    lc: &lincheck::LincheckClaim,
    x: &lincheck::QuirkyPoint,
) -> [ZClaim; 2] {
    [
        ZClaim {
            point: r.ab_claim_point(lc.r_inner_skip, &lc.r_inner_rest, &x.x_outer),
            value: lc.w,
        },
        ZClaim {
            point: r.c_claim_point(SkipPoint::Phi8(zc.z), &zc.r_rest),
            value: zc.c_eval,
        },
    ]
}
fn packed_bytes(v: &[F128]) -> &[u8] {
    assert!(cfg!(target_endian = "little"));
    // SAFETY: upstream F128 is repr(C, align(16)) with two public u64 fields,
    // no padding (16 bytes total), on a little-endian target. The view is shared
    // and cannot outlive the input. This is the canonical packed bit order.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// Inputs are canonical Flock row-major packed witness/a/b plus lincheck's stripe packing.
/// Witness generation is deliberately caller-owned and timed separately.
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
    let tower = z.iter().copied().map(from_flock).collect();
    let (cm, data) = pcs.commit(tower);
    timing.commit_ms = elapsed(t);
    let mut ch = FsChallenger::new(b"binary-hyperfold-flock");
    bind(&mut ch, r, pcs, &cm);
    let aa = packed_bytes(&a);
    let bb = packed_bytes(&b);
    let zz = packed_bytes(&z);
    let t = Instant::now();
    let (zp, zc) = zerocheck::prove_packed_padded(&aa, &bb, &zz, r.m, &r.padding_spec(), &mut ch);
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
    let mut proofs = vec![];
    let mut weights = vec![];
    for claim in &claims {
        let (p, out) = ring_switch::prove(&z, &outer(&claim.point), &mut ch);
        proofs.push(p);
        weights.push(
            out.rs_eq_ind
                .iter()
                .copied()
                .map(from_flock)
                .collect::<Vec<_>>(),
        );
    }
    // Combine the two independently reduced weighted sums before the separate sumcheck.
    let rho = from_flock(ch.sample_f128());
    let mut weight = weights.remove(0);
    for (x, y) in weight.iter_mut().zip(weights.remove(0)) {
        *x += rho * y;
    }
    let mut f = data.message.clone();
    let mut reduction = vec![];
    let mut point = vec![];
    for _ in 0..pcs.config.params.mu {
        let p = polynomial(&f, &weight, 0);
        ch.observe_f128_slice(&p.0.map(to_flock));
        let alpha = from_flock(ch.sample_f128());
        fold_table(&mut f, 0, alpha);
        fold_table(&mut weight, 0, alpha);
        point.push(alpha);
        reduction.push(p);
    }
    let packed_evaluation = f[0];
    ch.observe_f128(to_flock(packed_evaluation));
    timing.ring_switch_ms = elapsed(t);
    // A separate domain-separated HyperFold transcript binds cm, point and value.
    let t = Instant::now();
    let opening = pcs.open(&data, point, packed_evaluation);
    timing.open_ms = elapsed(t);
    (
        FlockHyperFoldProof {
            commitment: cm,
            zerocheck: zp,
            lincheck: lp,
            ring_switch: proofs,
            reduction,
            packed_evaluation,
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
    if !r.c0_is_identity()
        || r.m - 7 != pcs.config.params.mu
        || p.ring_switch.len() != 2
        || p.reduction.len() != pcs.config.params.mu
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
    let mut out = vec![];
    for (claim, rp) in claims.iter().zip(&p.ring_switch) {
        let full = outer(&claim.point);
        if rp.s_hat_v.len() != 128 {
            return false;
        }
        let Ok(v) = ring_switch::verify_succinct_with_grinding(
            claim.value,
            &claim.point.z_skip.weights(r.k_skip),
            &full,
            rp,
            0,
            &mut ch,
        ) else {
            return false;
        };
        out.push((full, v));
    }
    let rho = from_flock(ch.sample_f128());
    let mut sum = from_flock(out[0].1.sumcheck_claim) + rho * from_flock(out[1].1.sumcheck_claim);
    let mut point = vec![];
    for &poly in &p.reduction {
        if poly.sum() != sum {
            return false;
        }
        ch.observe_f128_slice(&poly.0.map(to_flock));
        let alpha = from_flock(ch.sample_f128());
        sum = poly.evaluate(alpha);
        point.push(alpha);
    }
    let query: Vec<_> = point.iter().copied().map(to_flock).collect();
    let mut weight = BF128::ZERO;
    let mut power = BF128::ONE;
    for (full, output) in &out {
        weight += power
            * from_flock(ring_switch::eval_rs_eq(
                &full[1..],
                &query,
                &output.eq_r_dprime,
            ));
        power *= rho;
    }
    if sum != p.packed_evaluation * weight {
        return false;
    }
    ch.observe_f128(to_flock(p.packed_evaluation));
    pcs.verify(&p.commitment, point, p.packed_evaluation, &p.opening)
        .is_ok()
}

/// The same staged Boolean PIOP with Flock's public Ligerito opener, for a disclosed
/// orchestration-level baseline. Ring-switch remains fused in its open_ms.
pub fn prove_r1cs_with_ligerito_timed(
    r: &BlockR1cs,
    params: &flock_core::pcs::PcsParams,
    z: Vec<F128>,
    a: Vec<F128>,
    b: Vec<F128>,
    zlin: &[u8],
    circuit: &dyn LincheckCircuit,
) -> (
    flock_core::proof::R1csProofLigerito,
    flock_core::pcs::Commitment,
    PhaseTimings,
) {
    let mut times = PhaseTimings::default();
    let mut ch = FsChallenger::new(b"binary-hyperfold-ligerito-baseline");
    let t = Instant::now();
    let (cm, data) = flock_core::pcs::commit(&z, params);
    times.commit_ms = elapsed(t);
    flock_core::proof::bind_statement(&mut ch, r, &cm);
    let aa = packed_bytes(&a);
    let bb = packed_bytes(&b);
    let zz = packed_bytes(&z);
    let t = Instant::now();
    let (zp, zc) = zerocheck::prove_packed_padded(&aa, &bb, &zz, r.m, &r.padding_spec(), &mut ch);
    times.zerocheck_ms = elapsed(t);
    let x = r.x_ab_from_multilinear_values(SkipPoint::Phi8(zc.z), &zc.mlv_challenges);
    let t = Instant::now();
    let (lp, lc, _) = lincheck::prove_padded_capture_z_vec(
        zlin,
        r.m,
        r.k_log,
        r.k_skip,
        r.useful_bits,
        circuit,
        &x,
        &mut ch,
    );
    times.lincheck_ms = elapsed(t);
    let claims = claims(r, &zc, &lc, &x);
    let full: Vec<_> = claims.iter().map(|c| outer(&c.point)).collect();
    let refs: Vec<_> = full.iter().map(Vec::as_slice).collect();
    let config = params
        .ligerito_prover_config()
        .expect("no embedded Ligerito configuration for this dimension");
    let t = Instant::now();
    let pcs_open = flock_core::pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding(
        z,
        &data,
        &cm,
        &refs,
        &[None, None],
        &[],
        &r.padding_spec(),
        &config,
        params.opening_grinding(),
        &mut ch,
    );
    times.open_ms = elapsed(t);
    (
        flock_core::proof::R1csProofLigerito {
            zerocheck: zp,
            lincheck: lp,
            pcs_open,
        },
        cm,
        times,
    )
}
