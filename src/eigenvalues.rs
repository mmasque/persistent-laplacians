use crate::{is_float_zero, to_dense, TOL};
use eigenvalues::{lanczos::HermitianLanczos, SpectrumTarget};
use lanczos::{Hermitian, Order};
use nalgebra_sparse::CsrMatrix;

/// Functions to calculate eigenvalues from persistent laplacian

pub fn empty(_persistent_laplacian: &CsrMatrix<f64>, _num_eigenvalues: usize) -> Vec<f64> {
    vec![]
}

pub fn compute_nonzero_eigenvalues_from_persistent_laplacian_dense(
    persistent_laplacian: &CsrMatrix<f64>,
    num_nonzero_eigenvalues: usize,
) -> Vec<f64> {
    assert!(persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0);
    let dense = to_dense(&persistent_laplacian);
    let mut nonzero_eigenvalues: Vec<f64> = dense
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .filter(|&&sigma| !is_float_zero(sigma))
        .cloned()
        .collect();
    nonzero_eigenvalues.truncate(num_nonzero_eigenvalues);
    nonzero_eigenvalues
}

pub fn compute_homology_from_persistent_laplacian_eigenvalues(
    persistent_laplacian: &CsrMatrix<f64>,
    num_nonzero_eigenvalues: usize,
) -> Vec<f64> {
    assert!(persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0);
    let dense = to_dense(&persistent_laplacian);
    let spectrum_target = SpectrumTarget::Lowest;
    let lanczos = HermitianLanczos::new(dense, 50, spectrum_target).unwrap();
    let mut nonzero_eigenvalues: Vec<f64> = lanczos
        .eigenvalues
        .iter()
        .filter(|&&sigma| !is_float_zero(sigma))
        .cloned()
        .collect();
    nonzero_eigenvalues.truncate(num_nonzero_eigenvalues);
    nonzero_eigenvalues
}

pub fn compute_eigenvalues_from_persistent_laplacian_lanczos_crate(
    persistent_laplacian: &CsrMatrix<f64>,
    num_nonzero_eigenvalues: usize,
) -> Vec<f64> {
    assert!(persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0);
    let eigen = persistent_laplacian.eigsh(50, Order::Smallest);
    let mut nonzero_eigenvalues: Vec<f64> = eigen
        .eigenvalues
        .iter()
        .filter(|&&sigma| !is_float_zero(sigma))
        .cloned()
        .collect();
    nonzero_eigenvalues.truncate(num_nonzero_eigenvalues);
    nonzero_eigenvalues
}

pub fn compute_eigenvalues_from_persistent_laplacian_primme_crate(
    persistent_laplacian: &CsrMatrix<f64>,
    num_nonzero_eigenvalues: usize,
) -> Vec<f64> {
    assert!(persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0);
    primme::smallest_nonzero_eigenvalues(persistent_laplacian, num_nonzero_eigenvalues, TOL)
        .unwrap_or(vec![])
}

#[cfg(test)]
mod tests {
    use nalgebra_sparse::{CooMatrix, CsrMatrix};
    use pyo3::Python;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::{
        eigenvalues::compute_eigenvalues_from_persistent_laplacian_primme_crate,
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
    #[test]
    fn test_python_vs_primme() {
        let matrix = generate_random_symmetric_matrix(500, 0.05);
        let primme_eigs = compute_eigenvalues_from_persistent_laplacian_primme_crate(&matrix, 1);
        let scipy_eigs = Python::with_gil(|py| {
            let scipy_config = ScipyEigshConfig::default_from_num_nonzero_eigenvalues(1, py);
            eigsh_scipy(&matrix, &scipy_config).unwrap()
        });
        println!("PRIMME: {:?}", primme_eigs);
        println!("SCIPY: {:?}", scipy_eigs);
    }
}
