use binary_hyperfold::{
    BF128,
    code::{
        HyperFoldableCode,
        folding::eq_weights,
        matrix_h::{LocalMatrix, MatrixKind},
    },
    parameters::{HyperFoldConfig, HyperFoldParams},
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use p3_binary_field::TowerLevel;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
fn bench(c: &mut Criterion) {
    for r in [2usize, 4, 8] {
        let h = LocalMatrix::new(r, MatrixKind::Specialized);
        c.bench_function(&format!("H{r}"), |b| {
            let mut x = vec![BF128::from_repr(77); r];
            b.iter(|| {
                h.apply(black_box(&mut x), false);
                black_box(&x);
            });
        });
        let conf = HyperFoldConfig::derive(&HyperFoldParams {
            mu: 8 + 2 * r.ilog2() as usize,
            mu_0: Some(8),
            r,
            ..Default::default()
        })
        .unwrap();
        let code = HyperFoldableCode::setup(&conf, &mut ChaCha8Rng::seed_from_u64(7));
        let input = vec![BF128::from_repr(123); conf.k[conf.d]];
        let word = code.encode(conf.d, &input);
        let beta = eq_weights(&vec![BF128::from_repr(34); conf.nu]);
        c.bench_function(&format!("Enc-r{r}"), |b| {
            b.iter(|| black_box(code.encode(conf.d, black_box(&input))))
        });
        c.bench_function(&format!("Fold-r{r}"), |b| {
            b.iter(|| black_box(code.fold(conf.d, black_box(&word), black_box(&beta))))
        });
    }
}
criterion_group!(benches, bench);
criterion_main!(benches);
