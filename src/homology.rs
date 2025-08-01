use lanczos::{Hermitian, Order};
use nalgebra_sparse::CsrMatrix;
use numpy::{IntoPyArray, PyArray1};
use pyo3::types::PyDict;
use pyo3::{prelude::*, types::PyList};

use crate::{is_float_zero, to_dense};

// Functions to calculate homology from persistent laplacian

pub fn compute_homology_from_persistent_laplacian_scipy(
    persistent_laplacian: &CsrMatrix<f64>,
    scipy_config: &ScipyEigshConfig,
    _zero_tol: f64,
) -> usize {
    let res = eigsh_scipy(persistent_laplacian, scipy_config);
    if persistent_laplacian.ncols() > scipy_config.k {
        if let Ok(eigs) = eigsh_scipy(persistent_laplacian, scipy_config) {
            eigs.iter()
                .filter(|x| is_float_zero(**x, scipy_config.tol))
                .count()
        } else {
            println!("Error in scipy routine: {:?}", res.err());
            compute_homology_from_persistent_laplacian_dense(persistent_laplacian, scipy_config.tol)
        }
    } else {
        compute_homology_from_persistent_laplacian_dense(persistent_laplacian, scipy_config.tol)
    }
}

pub fn compute_homology_from_persistent_laplacian_dense(
    persistent_laplacian: &CsrMatrix<f64>,
    zero_tol: f64,
) -> usize {
    assert!(persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0);
    let dense = to_dense(&persistent_laplacian);
    let svd = nalgebra_sparse::na::SVD::new(dense, false, false);
    let nullity = svd
        .singular_values
        .iter()
        .filter(|&&sigma| sigma.abs() < zero_tol)
        .count();
    nullity
}

pub fn compute_homology_from_persistent_laplacian_dense_eigen(
    persistent_laplacian: &CsrMatrix<f64>,
    zero_tol: f64,
) -> usize {
    assert!(persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0);
    let dense = to_dense(&persistent_laplacian);
    dense
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .filter(|x| is_float_zero(**x, zero_tol))
        .count()
}

// fn compute_homology_from_persistent_laplacian_eigenvalues(
//     persistent_laplacian: &CsrMatrix<f64>,
// ) -> usize {
//     assert!(persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0);
//     let dense = to_dense(&persistent_laplacian);
//     let spectrum_target = SpectrumTarget::Lowest;
//     let lanczos = HermitianLanczos::new(dense, 50, spectrum_target).unwrap();
//     lanczos
//         .eigenvalues
//         .iter()
//         .filter(|x| is_float_zero(**x))
//         .count()
// }

pub fn compute_homology_from_persistent_laplacian_lanczos_crate(
    persistent_laplacian: &CsrMatrix<f64>,
    zero_tol: f64,
) -> usize {
    assert!(persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0);
    let eigen = persistent_laplacian.eigsh(50, Order::Smallest);
    eigen
        .eigenvalues
        .iter()
        .filter(|x| is_float_zero(**x, zero_tol))
        .count()
}

pub fn count_nnz_persistent_laplacian(
    persistent_laplacian: &CsrMatrix<f64>,
    _zero_tol: f64,
) -> usize {
    assert!(persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0);
    persistent_laplacian.nnz()
}

pub struct ScipyEigshConfig<'a> {
    pub py: Python<'a>,
    pub k: usize,
    pub sigma: Option<f64>,
    pub tol: f64,
    pub maxiter: Option<usize>,
    pub which: &'a str,
    pub eigsh: &'a PyAny,
    pub scipy_sparse: &'a PyModule,
}

impl<'a> ScipyEigshConfig<'a> {
    fn new(
        py: Python<'a>,
        k: usize,
        sigma: Option<f64>,
        tol: f64,
        maxiter: Option<usize>,
        which: &'a str,
        eigsh: &'a PyAny,
        scipy_sparse: &'a PyModule,
    ) -> Self {
        ScipyEigshConfig {
            py,
            k,
            sigma,
            tol,
            maxiter,
            which,
            eigsh,
            scipy_sparse,
        }
    }
    pub fn new_from_num_nonzero_eigenvalues_tol(
        num_nonzero_eigenvalues: usize,
        tol: f64,
        py: Python,
    ) -> ScipyEigshConfig {
        let sys = py.import("sys").unwrap();
        let path: &PyList = sys.getattr("path").unwrap().downcast().unwrap();
        path.insert(0, ".venv/lib/python3.10/site-packages")
            .unwrap();
        let eigsh = PyModule::import(py, "scipy.sparse.linalg")
            .unwrap()
            .getattr("eigsh")
            .unwrap();
        let scipy_sparse = PyModule::import(py, "scipy.sparse").unwrap();
        ScipyEigshConfig::new(
            py,
            num_nonzero_eigenvalues,
            // TODO: make configurable
            Some(1e-2),
            tol,
            None,
            "LA",
            &eigsh,
            &scipy_sparse,
        )
    }
}

/// Tries to find nonzero eigenvalues as specified by the config, but only performs
/// one computation. Zero eigenvalues are filtered out, but if not enough nonzero eigenvalues remain,
/// the function returns fewer than `scipy_config.k` eigenvalues.
fn eigsh_scipy_inner(
    csr_matrix: &PyAny,
    scipy_config: &ScipyEigshConfig,
    kwargs: &PyDict,
) -> PyResult<Vec<f64>> {
    let result = scipy_config.eigsh.call((csr_matrix,), Some(kwargs))?;
    let eigvals_py: &PyArray1<f64> = result.get_item(0)?.downcast()?;
    let mut eigvals = unsafe { eigvals_py.as_slice()?.to_vec() };
    eigvals = eigvals
        .iter()
        .filter(|x| !is_float_zero(**x, scipy_config.tol))
        .cloned()
        .collect();
    Ok(eigvals)
}

pub fn eigsh_scipy(a: &CsrMatrix<f64>, scipy_config: &ScipyEigshConfig) -> PyResult<Vec<f64>> {
    // If matrix is zero, exit early
    if a.nnz() == 0
        || a.values()
            .iter()
            .filter(|x| !is_float_zero(**x, scipy_config.tol))
            .count()
            == 0
    {
        return Ok(vec![]);
    }
    let (nrows, ncols) = (a.nrows(), a.ncols());
    let indptr: Vec<usize> = a.row_offsets().iter().cloned().collect();
    let indices: Vec<usize> = a.col_indices().iter().cloned().collect();
    let data: Vec<f64> = a.values().iter().cloned().collect();

    let py = scipy_config.py.clone();
    let py_indptr = indptr.into_pyarray(py);
    let py_indices = indices.into_pyarray(py);
    let py_data = data.into_pyarray(py);

    let csr_matrix = scipy_config
        .scipy_sparse
        .getattr("csr_matrix")?
        .call1(((py_data, py_indices, py_indptr), (nrows, ncols)))?;

    let kwargs = PyDict::new(scipy_config.py.clone());

    kwargs.set_item("which", scipy_config.which)?;
    if let Some(s) = scipy_config.sigma {
        kwargs.set_item("sigma", s)?;
    }
    kwargs.set_item("tol", scipy_config.tol)?;
    if let Some(mi) = scipy_config.maxiter {
        kwargs.set_item("maxiter", mi)?;
    }

    // Compute more eigenvalues than we need, throw away zero eigenvalues, and recompute if needed.
    let mut k = nrows.min(2 * scipy_config.k);
    let mut nonzero_eigvals = eigsh_scipy_inner(csr_matrix, scipy_config, kwargs)?;
    let num_obtained_nonzero_eigenvalues = nonzero_eigvals.len();
    while num_obtained_nonzero_eigenvalues < scipy_config.k && k < nrows {
        // Increase k
        k = nrows.min(k * 2);
        kwargs.set_item("k", k)?;
        // Compute more eigenvalues
        nonzero_eigvals = eigsh_scipy_inner(csr_matrix, scipy_config, kwargs)?;
    }
    nonzero_eigvals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    nonzero_eigvals = nonzero_eigvals.into_iter().take(scipy_config.k).collect();
    Ok(nonzero_eigvals)
}
