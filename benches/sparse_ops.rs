// Compare different sparse implementations
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra_sparse::{coo::CooMatrix, CsrMatrix};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use sprs::TriMat;

const N: usize = 1_000;
const NNZ: usize = 5_000;

fn make_nalgebra() -> CsrMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut coo = CooMatrix::new(N, N);
    for _ in 0..NNZ {
        let i = (rng.next_u32() as usize) % N;
        let j = (rng.next_u32() as usize) % N;
        let v = f64::from_bits(rng.next_u64());
        coo.push(i, j, v);
    }
    CsrMatrix::from(&coo)
}

fn make_sprs() -> sprs::CsMat<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut tri = TriMat::with_capacity((N, N), NNZ);
    for _ in 0..NNZ {
        let i = (rng.next_u32() as usize) % N;
        let j = (rng.next_u32() as usize) % N;
        let v = f64::from_bits(rng.next_u64());
        tri.add_triplet(i, j, v);
    }
    tri.to_csr()
}

fn bench_sparse(c: &mut Criterion) {
    let a_na = make_nalgebra();
    let b_na = make_nalgebra();
    let a_sp = make_sprs();
    let b_sp = make_sprs();

    let mut group = c.benchmark_group("sparse_ops");

    // nalgebra add
    group.bench_function(BenchmarkId::new("nalgebra", "add"), |b| {
        b.iter(|| &a_na + &b_na)
    });
    // nalgebra sub
    group.bench_function(BenchmarkId::new("nalgebra", "sub"), |b| {
        b.iter(|| &a_na - &b_na)
    });
    // nalgebra mul
    group.bench_function(BenchmarkId::new("nalgebra", "mul"), |b| {
        b.iter(|| &a_na * &b_na)
    });

    // sprs add
    group.bench_function(BenchmarkId::new("sprs", "add"), |b| {
        b.iter(|| &a_sp + &b_sp)
    });
    // sprs sub
    group.bench_function(BenchmarkId::new("sprs", "sub"), |b| {
        b.iter(|| &a_sp - &b_sp)
    });
    // sprs mul
    group.bench_function(BenchmarkId::new("sprs", "mul"), |b| {
        b.iter(|| &a_sp * &b_sp)
    });

    group.finish();
}

criterion_group!(benches, bench_sparse);
criterion_main!(benches);
