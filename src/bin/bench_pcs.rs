use binary_hyperfold::{
    BF128,
    parameters::{DistanceMode, HyperFoldConfig, HyperFoldParams, MatrixKind},
    pcs::BinaryHyperFoldPcs,
    sumcheck,
};
use clap::Parser;
use p3_binary_field::TowerLevel;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{
    fs::File,
    io::{self, Write},
    time::Instant,
};
#[derive(Parser)]
struct Args {
    #[arg(long, value_delimiter = ',', default_value = "20")]
    mu: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "2,4,8")]
    r: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "2")]
    c: Vec<usize>,
    #[arg(long, default_value_t = 128)]
    base_field_bits: usize,
    #[arg(long)]
    mu0: Option<usize>,
    #[arg(long, default_value_t = 100)]
    security_bits: usize,
    #[arg(long, default_value = "analytic")]
    distance_mode: String,
    #[arg(long, default_value = "specialized")]
    matrix: String,
    #[arg(long, default_value_t = 32)]
    threads: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 3)]
    iters: usize,
    #[arg(long)]
    csv: Option<String>,
    #[arg(long)]
    schedule_json: Option<String>,
}
fn main() {
    let a = Args::parse();
    rayon::ThreadPoolBuilder::new()
        .num_threads(a.threads)
        .build_global()
        .unwrap();
    let mut output: Box<dyn Write> = if let Some(p) = &a.csv {
        Box::new(File::create(p).unwrap())
    } else {
        Box::new(io::stdout())
    };
    writeln!(output,"mu,r,c,base_field_bits,security_bits,d,mu0,queries,distance,eta,delta,trial,commit_ms,open_ms,verify_ms,proof_bytes,wire_proof_bytes").unwrap();
    let mut schedules = Vec::new();
    for &mu in &a.mu {
        for &r in &a.r {
            for &c in &a.c {
                let p = HyperFoldParams {
                    mu,
                    r,
                    c,
                    security_bits: a.security_bits,
                    base_field_bits: a.base_field_bits,
                    mu_0: a.mu0,
                    distance_mode: match a.distance_mode.as_str() {
                        "analytic" => DistanceMode::Analytic,
                        "primal-dual" => DistanceMode::PrimalDual,
                        _ => panic!("distance mode"),
                    },
                    matrix_kind: match a.matrix.as_str() {
                        "specialized" => MatrixKind::Specialized,
                        "general" => MatrixKind::General,
                        _ => panic!("matrix"),
                    },
                    ..Default::default()
                };
                let config = HyperFoldConfig::derive(&p).unwrap();
                eprintln!(
                    "derived schedule (float search, not a numerical certificate): {config:?}"
                );
                schedules.push(config.clone());
                if a.iters == 0 {
                    continue;
                }
                let pcs = BinaryHyperFoldPcs::setup(config.clone(), [42; 32]);
                let mut rng = ChaCha8Rng::seed_from_u64(8);
                let message: Vec<_> = (0..1usize << mu)
                    .map(|_| {
                        BF128::from_repr(
                            rng.random::<u128>() & (u128::MAX >> (128 - a.base_field_bits)),
                        )
                    })
                    .collect();
                let point: Vec<_> = (0..mu).map(|_| BF128::from_repr(rng.random())).collect();
                let value = sumcheck::evaluate(&message, &point);
                for trial in 0..a.warmup + a.iters {
                    let owned = message.clone();
                    let start = Instant::now();
                    let (cm, data) = pcs.commit(owned);
                    let commit = start.elapsed().as_secs_f64() * 1000.0;
                    let start = Instant::now();
                    let proof = pcs.open(&data, point.clone(), value);
                    let open = start.elapsed().as_secs_f64() * 1000.0;
                    let start = Instant::now();
                    pcs.verify(&cm, point.clone(), value, &proof).unwrap();
                    let verify = start.elapsed().as_secs_f64() * 1000.0;
                    if trial >= a.warmup {
                        writeln!(output,"{mu},{r},{c},{},{},{},{},{},{},{},{},{},{commit:.6},{open:.6},{verify:.6},{},{}",a.base_field_bits,a.security_bits,config.d,config.mu_0,config.queries,config.distance,config.eta,config.delta,trial-a.warmup,cm.size_in_bytes()+proof.size_in_bytes(),bincode::serialized_size(&cm).unwrap() as usize+proof.wire_size_in_bytes()).unwrap();
                    }
                }
            }
        }
    }
    if let Some(path) = a.schedule_json {
        serde_json::to_writer_pretty(File::create(path).unwrap(), &schedules).unwrap();
    }
}
