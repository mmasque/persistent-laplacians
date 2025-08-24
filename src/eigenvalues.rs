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
        eigenvalues::{
            compute_nonzero_eigenvalues_from_persistent_laplacian_dense,
            compute_nonzero_eigenvalues_from_persistent_laplacian_scipy,
        },
        homology::ScipyEigshConfig,
    };

    pub fn generate_random_symmetric_psd_matrix(
        n: usize,
        density: f64,
        seed: u64,
    ) -> CsrMatrix<f64> {
        assert!((0.0..=1.0).contains(&density));
        let mut rng = StdRng::seed_from_u64(seed);
        // Build a random sparse A (n×n) in COO form.
        let mut a = CooMatrix::new(n, n);
        for i in 0..n {
            for j in 0..n {
                if rng.random::<f64>() < density {
                    let v = rng.random_range(-1.0..1.0);
                    if v != 0.0 {
                        a.push(i, j, v);
                    }
                }
            }
        }

        // Bucket A's entries by column: col -> Vec<(row, val)>
        let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (i, j, v) in a.triplet_iter() {
            cols[j].push((i, *v));
        }

        // Build ATA in COO by summing outer products of each column vector.
        let mut ata = CooMatrix::new(n, n);
        for col_entries in cols {
            for &(r1, v1) in &col_entries {
                for &(r2, v2) in &col_entries {
                    // Add both (r1,r2) contributions; COO will sum duplicates.
                    ata.push(r1, r2, v1 * v2);
                }
            }
        }

        CsrMatrix::from(&ata)
    }

    fn assert_close_vec(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(
            a.len(),
            b.len(),
            "lengths differ: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let ok = if x.is_finite() && y.is_finite() {
                (x - y).abs() <= tol
            } else {
                x == y // NaN/Inf: require exact match
            };
            assert!(ok, "idx {}: {} != {} (tol={})", i, x, y, tol);
        }
    }

    #[test]
    fn dense_vs_scipy_nonzero_eigs_match_on_random_mats() {
        // deterministic randomness
        let mut rng = StdRng::seed_from_u64(42);

        // test parameters
        let sizes = [10usize, 24, 37];
        let densities = [0.15, 0.25]; // edge probabilities for sparsity
        let reps = 3usize; // matrices per (n, d)
        let zero_tol = 1e-6;
        let cmp_tol = 1e-3;
        Python::with_gil(|py| {
            for &n in &sizes {
                let k = 1;

                for &d in &densities {
                    for _ in 0..reps {
                        // If your helper requires its own RNG, adapt accordingly.
                        let m: CsrMatrix<f64> = generate_random_symmetric_psd_matrix(n, d, 42);

                        // Dense path
                        let mut e_dense =
                            compute_nonzero_eigenvalues_from_persistent_laplacian_dense(
                                &m, k, zero_tol,
                            );

                        // SciPy path (adjust constructor if your API differs)
                        let scipy_cfg =
                            ScipyEigshConfig::new_from_num_nonzero_eigenvalues_tol(k, zero_tol, py);
                        let mut e_scipy =
                            compute_nonzero_eigenvalues_from_persistent_laplacian_scipy(
                                &m, &scipy_cfg,
                            );
                        println!("SCIPY: {:?}", e_scipy);
                        println!("DENSE: {:?}", e_dense);
                        // Sort before compare (eigensolvers may return in different orders)
                        e_dense.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        e_scipy.sort_by(|a, b| a.partial_cmp(b).unwrap());

                        // Compare with tolerance
                        assert_close_vec(&e_dense, &e_scipy, cmp_tol);
                    }
                }
            }
        });
    }
}
