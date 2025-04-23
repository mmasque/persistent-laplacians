use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra_sparse::coo::CooMatrix;
use rand::{Rng, SeedableRng};

use persistent_laplacians::{up_persistent_laplacian, SparseMatrix};
use rand_chacha::ChaCha8Rng;

fn generate_large_sparse_matrix(n: usize, nnz: usize) -> SparseMatrix<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(42); // Fixed seed
    let mut coo = CooMatrix::new(n, n);
    for _ in 0..nnz {
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        let val = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        coo.push(i, j, val);
    }
    SparseMatrix::from(coo)
}

fn bench_up_persistent_laplacian(c: &mut Criterion) {
    let sizes = vec![100, 200, 300, 400, 500, 1000];

    for &n in &sizes {
        let matrix = generate_large_sparse_matrix(n, 2 * n);

        c.bench_function(&format!("up_persistent_laplacian n={}", n), |b| {
            b.iter(|| {
                let _ = up_persistent_laplacian(&matrix, n - 1);
            });
        });
    }
}

criterion_group!(benches, bench_up_persistent_laplacian);
criterion_main!(benches);
