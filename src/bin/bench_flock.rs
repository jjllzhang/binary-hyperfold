use binary_hyperfold::{
    adapters::flock::*,
    parameters::{HyperFoldConfig, HyperFoldParams},
    pcs::BinaryHyperFoldPcs,
};
use clap::Parser;
use flock_core::{
    lincheck::LincheckCircuit,
    pcs::{
        PcsParams,
        ligerito::{LigeritoProfile, embedded_initial_k_or_default},
    },
    r1cs::BlockR1cs,
};
use flock_field::F128;
use flock_transcript::challenger::FsChallenger;
use std::time::Instant;
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "blake3")]
    hash: String,
    #[arg(long, default_value_t = 10)]
    log2_batches: usize,
    #[arg(long, value_delimiter = ',', default_value = "4")]
    r: Vec<usize>,
    #[arg(long, default_value_t = 32)]
    threads: usize,
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    #[arg(long, default_value_t = 1)]
    iters: usize,
}
fn main() {
    let args = Args::parse();
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap();
    assert!(flock_parallel::init_all_core_pool(args.threads));
    // Flock has optional all-core pools; keep the measured paths on the requested pool.
    // These knobs are read by the upstream implementation, not changed inside timed loops.
    let count = 1usize << args.log2_batches;
    let (r, z, a, b, zlin, witness_ms): (BlockR1cs, Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>, f64) =
        match args.hash.as_str() {
            "blake3" => {
                use flock_prover::r1cs_hashes::blake3;
                let blocks = vec![([0; 8], [0; 16], 0, 64, 0); count];
                let r = blake3::build_block_r1cs(args.log2_batches);
                let t = Instant::now();
                let (z, a, b, lin) = blake3::generate_witness_with_ab_packed_and_lincheck(
                    &blocks,
                    args.log2_batches,
                );
                let witness = t.elapsed().as_secs_f64() * 1000.0;
                (r, z, a, b, lin, witness)
            }
            "sha256" => {
                use flock_prover::r1cs_hashes::sha2;
                let blocks = vec![([0; 8], [0; 16]); count];
                let r = sha2::build_block_r1cs(args.log2_batches);
                let t = Instant::now();
                let (z, a, b, lin) =
                    sha2::generate_witness_with_ab_packed_and_lincheck(&blocks, args.log2_batches);
                let witness = t.elapsed().as_secs_f64() * 1000.0;
                (r, z, a, b, lin, witness)
            }
            _ => panic!("hash must be blake3 or sha256"),
        };
    let _ = r.statement_digest();
    eprintln!(
        "witness_ms is fast witness/a/b/lincheck packing, excludes R1CS construction; total_prove_ms includes this amortized once-per-configuration measurement. Ligerito Fast100 versus HyperFold PCS target100; not a full-SNARK security certificate."
    );
    let circuit = r.csc_lincheck_circuit();
    println!(
        "backend,hash,log2_batches,packed_mu,r,security_policy,trial,witness_ms,commit_ms,zerocheck_ms,lincheck_ms,ring_switch_ms,open_ms,total_prove_ms,k_hashes_per_s,verify_ms,proof_kb"
    );
    for &arity in &args.r {
        let config = HyperFoldConfig::derive(&HyperFoldParams {
            mu: r.m - 7,
            r: arity,
            ..Default::default()
        })
        .unwrap();
        let pcs = BinaryHyperFoldPcs::setup(config, [44; 32]);
        for trial in 0..args.warmup + args.iters {
            let input = (z.clone(), a.clone(), b.clone());
            let start = Instant::now();
            let (p, t) = prove_r1cs_with_binary_hyperfold(
                &r, &pcs, input.0, input.1, input.2, &zlin, circuit,
            );
            let total = witness_ms + start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            assert!(verify_r1cs_with_binary_hyperfold(&r, &pcs, &p, circuit));
            let verify = start.elapsed().as_secs_f64() * 1000.0;
            if trial >= args.warmup {
                println!(
                    "hyperfold,{},{},{},{arity},PCS-target100-PIOP-no-grind,{},{witness_ms:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{total:.6},{:.6},{verify:.6},{:.3}",
                    args.hash,
                    args.log2_batches,
                    r.m - 7,
                    trial - args.warmup,
                    t.commit_ms,
                    t.zerocheck_ms,
                    t.lincheck_ms,
                    t.ring_switch_ms,
                    t.open_ms,
                    count as f64 / total,
                    p.wire_size_in_bytes() as f64 / 1000.0
                );
            }
        }
    }
    baseline(&args, &r, &z, &a, &b, &zlin, circuit, witness_ms);
}
fn baseline(
    args: &Args,
    r: &BlockR1cs,
    z: &[F128],
    a: &[F128],
    b: &[F128],
    lin: &[u8],
    circuit: &dyn LincheckCircuit,
    witness_ms: f64,
) {
    let profile = LigeritoProfile::Fast100;
    let params = PcsParams {
        m: r.m,
        log_inv_rate: 1,
        log_batch_size: embedded_initial_k_or_default(r.m, profile),
        profile,
        num_lanes: None,
        merkle_hash: flock_hash::HashKind::Blake3,
    };
    for trial in 0..args.warmup + args.iters {
        let input = (z.to_vec(), a.to_vec(), b.to_vec());
        let start = Instant::now();
        let (proof, cm, t) =
            prove_r1cs_with_ligerito_timed(r, &params, input.0, input.1, input.2, lin, circuit);
        let total = witness_ms + start.elapsed().as_secs_f64() * 1000.0;
        let mut ch = FsChallenger::new(b"binary-hyperfold-ligerito-baseline");
        let start = Instant::now();
        flock_core::verifier::verify_ligerito(r, &cm, &proof, circuit, &params, &mut ch).unwrap();
        let verify = start.elapsed().as_secs_f64() * 1000.0;
        if trial >= args.warmup {
            println!(
                "ligerito,{},{},{},2,Fast100,{},{witness_ms:.6},{:.6},{:.6},{:.6},fused-in-open,{:.6},{total:.6},{:.6},{verify:.6},{:.3}",
                args.hash,
                args.log2_batches,
                r.m - 7,
                trial - args.warmup,
                t.commit_ms,
                t.zerocheck_ms,
                t.lincheck_ms,
                t.open_ms,
                (1usize << args.log2_batches) as f64 / total,
                bincode::serialized_size(&(&cm, &proof)).unwrap() as f64 / 1000.0
            );
        }
    }
}
