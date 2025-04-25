use nalgebra_sparse::coo::CooMatrix;
use rand::{Rng, SeedableRng};

use persistent_laplacians::SparseMatrix;
use rand_chacha::ChaCha8Rng;

pub fn generate_large_sparse_matrix(nrows: usize, ncols: usize, nnz: usize) -> SparseMatrix<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(42); // Fixed seed
    let mut coo = CooMatrix::new(nrows, ncols);
    for _ in 0..nnz {
        let i = rng.random_range(0..nrows);
        let j = rng.random_range(0..ncols);
        let val = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        coo.push(i, j, val);
    }
    SparseMatrix::from(coo)
}
