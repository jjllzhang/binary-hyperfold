//! Flock's Boolean PIOP and ring-switch with a weighted HyperFold opening.
use crate::{
    BF128,
    pcs::{BinaryHyperFoldPcs, Commitment},
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
use std::time::Instant;

#[path = "flock_weighted.rs"]
mod weighted;
pub use weighted::{
    FlockHyperFoldProof, prove_r1cs_with_binary_hyperfold, verify_r1cs_with_binary_hyperfold,
};

pub fn from_flock(x: F128) -> BF128 {
    Ghash128::from_repr(x.lo as u128 | ((x.hi as u128) << 64)).into()
}
pub fn to_flock(x: BF128) -> F128 {
    let v = Ghash128::from(x).to_repr();
    F128::new(v as u64, (v >> 64) as u64)
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
    ch.observe_label(b"flock-binary-hyperfold-weighted-v1");
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
