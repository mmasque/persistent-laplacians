use crate::{
    homology::{eigsh_scipy, ScipyEigshConfig},
    is_float_zero, to_dense,
};
use lanczos::{Hermitian, Order};
use nalgebra_sparse::CsrMatrix;

/// Functions to calculate eigenvalues from persistent laplacian

pub fn empty(
    _persistent_laplacian: &CsrMatrix<f64>,
    _num_eigenvalues: usize,
    _zero_tol: f64,
) -> Vec<f64> {
    vec![]
}

pub fn compute_nonzero_eigenvalues_from_persistent_laplacian_dense(
    persistent_laplacian: &CsrMatrix<f64>,
    num_nonzero_eigenvalues: usize,
    zero_tol: f64,
) -> Vec<f64> {
    assert!(persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0);
    let dense = to_dense(&persistent_laplacian);
    let mut nonzero_eigenvalues: Vec<f64> = dense
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .filter(|&&sigma| !is_float_zero(sigma, zero_tol))
        .cloned()
        .collect();
    nonzero_eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
    nonzero_eigenvalues.truncate(num_nonzero_eigenvalues);
    nonzero_eigenvalues
}

pub fn compute_nonzero_eigenvalues_from_persistent_laplacian_scipy(
    persistent_laplacian: &CsrMatrix<f64>,
    scipy_config: &ScipyEigshConfig,
) -> Vec<f64> {
    match eigsh_scipy(persistent_laplacian, scipy_config) {
        Ok(eigs) => eigs,
        Err(err) => {
            println!(
                "Error in scipy routine, falling back to dense computation: {:?}",
                err
            );
            compute_nonzero_eigenvalues_from_persistent_laplacian_dense(
                persistent_laplacian,
                scipy_config.k,
                scipy_config.tol,
            )
        }
    }
}

pub fn compute_eigenvalues_from_persistent_laplacian_lanczos_crate(
    persistent_laplacian: &CsrMatrix<f64>,
    num_nonzero_eigenvalues: usize,
    zero_tol: f64,
) -> Vec<f64> {
    assert!(persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0);
    let eigen = persistent_laplacian.eigsh(50, Order::Smallest);
    let mut nonzero_eigenvalues: Vec<f64> = eigen
        .eigenvalues
        .iter()
        .filter(|&&sigma| !is_float_zero(sigma, zero_tol))
        .cloned()
        .collect();
    nonzero_eigenvalues.truncate(num_nonzero_eigenvalues);
    nonzero_eigenvalues
}

// pub fn compute_eigenvalues_from_persistent_laplacian_primme_crate(
//     persistent_laplacian: &CsrMatrix<f64>,
//     num_nonzero_eigenvalues: usize,
//     zero_tol: f64,
// ) -> Vec<f64> {
//     assert!(persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0);
//     primme::smallest_nonzero_eigenvalues(persistent_laplacian, num_nonzero_eigenvalues, zero_tol)
//         .unwrap_or(vec![])
// }

#[cfg(test)]
mod tests {
    use nalgebra_sparse::{CooMatrix, CsrMatrix};
    use pyo3::Python;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::{
        // eigenvalues::compute_eigenvalues_from_persistent_laplacian_primme_crate,
        homology::{eigsh_scipy, ScipyEigshConfig},
    };
    fn generate_random_symmetric_matrix(n: usize, d: f64) -> CsrMatrix<f64> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut coo = CooMatrix::zeros(n, n);
        // Generate symmetric sparse matrix
        for i in 0..n {
            for j in i..n {
                if rng.random_bool(d) {
                    let val: f64 = rng.sample(rand::distr::StandardUniform);
                    coo.push(i, j, val);
                    if i != j {
                        coo.push(j, i, val);
                    }
                }
            }
        }
        let matrix = CsrMatrix::from(&coo);
        matrix
    }
}
