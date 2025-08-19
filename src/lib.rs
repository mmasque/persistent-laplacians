use csv::Writer;
use dashmap::DashMap;
use homology::compute_homology_from_persistent_laplacian_dense;
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::CooMatrix;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::time::Instant;

use crate::eigenvalues::{
    compute_nonzero_eigenvalues_from_persistent_laplacian_dense,
    compute_nonzero_eigenvalues_from_persistent_laplacian_scipy,
};
// use crate::eigenvalues::compute_eigenvalues_from_persistent_laplacian_primme_crate;
use crate::homology::{eigsh_scipy, ScipyEigshConfig};
use crate::laplacians::{
    compute_down_persistent_laplacian_transposing, compute_up_persistent_laplacian_schur,
    compute_up_persistent_laplacian_stepwise, up_laplacian_transposing,
};
use crate::sparse::*;
use crate::utils::{is_float_zero, upper_submatrix_csr};

pub mod eigenvalues;
pub mod homology;
pub mod laplacians;
pub mod sparse;

mod utils;

/// Parses a nested dictionary from Python into a HashMap of HashMaps.
/// `filt` is expected to be a dictionary where keys are integers
pub fn parse_nested_dict(filt: &PyDict) -> PyResult<HashMap<usize, HashMap<usize, usize>>> {
    let mut filt_map = HashMap::with_capacity(filt.len());

    for (k_obj, sub_obj) in filt.iter() {
        let t: usize = k_obj.extract()?;
        let sub_dict: &PyDict = sub_obj.downcast::<PyDict>()?;
        let mut inner_map = HashMap::with_capacity(sub_dict.len());

        for (dim_obj, idx_obj) in sub_dict.iter() {
            let dim: usize = dim_obj.extract()?;
            let idx: usize = idx_obj.extract()?;
            inner_map.insert(dim, idx);
        }

        filt_map.insert(t, inner_map);
    }

    Ok(filt_map)
}

pub fn process_sparse_dict(dict: &PyDict) -> PyResult<HashMap<usize, SparseMatrix<f64>>> {
    let mut result = HashMap::new();
    for (key, value) in dict.iter() {
        let key: usize = key.extract()?;
        let n_rows: usize = value.get_item("n_rows")?.extract()?;
        let n_cols: usize = value.get_item("n_cols")?.extract()?;

        let cols_pyarray: &PyArray1<i64> = value.get_item("cols")?.downcast()?;
        let cols_readonly = cols_pyarray.readonly();
        let cols = cols_readonly.as_slice()?;

        let rows_pyarray: &PyArray1<i64> = value.get_item("rows")?.downcast()?;
        let rows_readonly = rows_pyarray.readonly();
        let rows = rows_readonly.as_slice()?;

        let data: &PyArray1<f64> = value.get_item("data")?.downcast()?;
        let data_readonly = data.readonly();
        let data: &[f64] = data_readonly.as_slice()?;

        let triplet_iter = rows
            .iter()
            .zip(cols.iter())
            .zip(data.iter())
            .map(|((&r, &c), &v)| (r.try_into().unwrap(), c.try_into().unwrap(), v));

        let coo = nalgebra_sparse::CooMatrix::try_from_triplets_iter(n_rows, n_cols, triplet_iter)
            .unwrap();

        result.insert(key, SparseMatrix::from(coo));
    }
    Ok(result)
}

/// Computes persistent homology of a filtration using persistent laplacians.
/// Assumes number of (q) simplices increases by at most 1 on each step of filtration.
///
/// `sparse_boundary_maps` is a map from dimension to sparse boundary maps.
/// `filt_hash` is a map from filtration index to a map of dimension to number of simplices.
/// `compute_homology_from_persistent_laplacian` is a function that computes homology from a persistent laplacian.
/// `zero_tol` is a tolerance for considering a floating-point number as zero.
pub fn persistent_homology_of_filtration<F>(
    sparse_boundary_maps: HashMap<usize, SparseMatrix<f64>>,
    filt_hash: HashMap<usize, HashMap<usize, usize>>,
    compute_homology_from_persistent_laplacian: F,
    zero_tol: f64,
) -> HashMap<usize, HashMap<(usize, usize), usize>>
where
    F: Fn(&CsrMatrix<f64>, f64) -> usize,
{
    // q: {(K, L): eigenvalues of pair K \hookrightarrow L}
    let mut homologies = HashMap::new();

    let mut filtration_indices: Vec<_> = filt_hash.keys().copied().collect();
    filtration_indices.sort_by(|a, b| b.cmp(a));
    let mut dimensions: Vec<&usize> = sparse_boundary_maps.keys().collect::<Vec<_>>();
    dimensions.sort();

    // dimension hashmap: q: dim of C_q at filtration indices
    for q in dimensions {
        let global_boundary_map_qm1 = match sparse_boundary_maps.get(&(q - 1)) {
            Some(map) => &map,
            None => &SparseMatrix::from(CooMatrix::zeros(0, 0)),
        };
        let num_qm1_simplices_global = global_boundary_map_qm1.csc.ncols();
        let global_boundary_map_q = match sparse_boundary_maps.get(&q) {
            Some(map) => &map,
            None => &SparseMatrix::from(CooMatrix::zeros(num_qm1_simplices_global, 0)),
        };
        let num_q_simplices_global = global_boundary_map_q.csc.ncols();
        let global_boundary_map_qp1 = match sparse_boundary_maps.get(&(q + 1)) {
            Some(map) => &map,
            None => &SparseMatrix::from(CooMatrix::zeros(num_q_simplices_global, 0)),
        };
        // Initialize results hash of eigenvalues for q-simplices.
        let q_homologies = DashMap::new();
        for l in &filtration_indices {
            let dimension_hashmap_l = filt_hash.get(l).unwrap();
            let num_q_simplices_l = dimension_hashmap_l.get(&q).unwrap();
            // Only have a boundar y map if there are higher dimensional simplices
            let num_qp1_simplices_l = dimension_hashmap_l.get(&(q + 1)).unwrap_or(&0);
            let boundary_map_l_qp1: Option<CsrMatrix<f64>> = if num_qp1_simplices_l > &0 {
                Some(
                    upper_submatrix_csr(
                        &global_boundary_map_qp1.csr,
                        *num_q_simplices_l,
                        *num_qp1_simplices_l,
                        zero_tol,
                    )
                    .into(),
                )
            } else {
                None
            };
            let mut up_persistent_laplacian =
                boundary_map_l_qp1.map(|b| up_laplacian_transposing(&b));
            // For each filtration value lower than the current filtration, compute the persistent laplacian.
            for k in (0..=*l).rev() {
                let dimension_hashmap_k = filt_hash.get(&k).unwrap();
                let num_q_simplices_k = dimension_hashmap_k.get(&q).unwrap();
                if num_q_simplices_k == &0 {
                    // The map is always zero, so we don't care
                    continue;
                }
                let num_qm1_simplices_k = dimension_hashmap_k.get(&(q - 1)).unwrap_or(&0);
                // Compute the up persistent laplacian for K \hookrightarrow L inductively
                up_persistent_laplacian = up_persistent_laplacian.and_then(|u| {
                    compute_up_persistent_laplacian_stepwise(*num_q_simplices_k, u, zero_tol)
                });
                // Compute the down persistent laplacian for K \hookrightarrow L
                // If there are no lower simplices, the map factors via the 0 vector space, so it is zero
                let down_persistent_laplacian = if num_qm1_simplices_k > &0 {
                    Some(compute_down_persistent_laplacian_transposing(
                        *num_qm1_simplices_k,
                        *num_q_simplices_k,
                        &global_boundary_map_q.csr,
                        zero_tol,
                    ))
                } else {
                    None
                };

                if let Some(persistent_laplacian) =
                    match (&up_persistent_laplacian, &down_persistent_laplacian) {
                        (Some(up), Some(down)) => Some(up + down),
                        (None, None) => None,
                        (Some(up), None) => Some(up.clone()),
                        (None, Some(down)) => Some(down.clone()),
                    }
                {
                    let homology =
                        compute_homology_from_persistent_laplacian(&persistent_laplacian, zero_tol);

                    // We don't need to compute for lower K if persistent homology is zero for this pair
                    if homology == 0 {
                        break;
                    }
                    q_homologies.insert((k, *l), homology);
                }
            }
        }
        homologies.insert(*q, q_homologies.into_iter().collect());
    }
    homologies
}

/// Computes some nonzero persistent eigenvalues of a filtration.
///
/// `sparse_boundary_maps` is a map from dimension to sparse boundary maps
/// `filt_hash` is a map from filtration index to a map of dimension to number of simplices
/// `compute_up_persistent_laplacian` is a function that computes the up persistent laplacian
/// `compute_nonzero_eigenvalues` is a function that computes nonzero eigenvalues from a persistent laplacian
/// `num_nonzero_eigenvalues` is the number of nonzero eigenvalues to compute
/// `downsampled_filtration_indices` is an optional vector of indices of the filtration to use when downsampling
/// `zero_tol` is a tolerance for considering a floating-point number as zero
pub fn persistent_eigenvalues_of_filtration<F, G>(
    sparse_boundary_maps: HashMap<usize, SparseMatrix<f64>>,
    // filtration_index: {q: dimension_q_simplices}
    filt_hash: HashMap<usize, HashMap<usize, usize>>,
    compute_up_persistent_laplacian: F,
    compute_nonzero_eigenvalues: G,
    num_nonzero_eigenvalues: usize,
    // Indices of the filtration to use when downsampling
    downsampled_filtration_indices: Option<Vec<usize>>,
    zero_tol: f64,
) -> HashMap<usize, HashMap<(usize, usize), Vec<f64>>>
where
    F: Fn(usize, CsrMatrix<f64>, f64) -> Option<CsrMatrix<f64>>,
    G: Fn(&CsrMatrix<f64>, usize, f64) -> Vec<f64>,
{
    // q: {(K, L): eigenvalues of pair K \hookrightarrow L}
    let mut eigenvalues = HashMap::new();

    let mut filtration_indices =
        downsampled_filtration_indices.unwrap_or_else(|| filt_hash.keys().copied().collect());
    filtration_indices.sort_by(|a, b| b.cmp(a));
    let mut dimensions: Vec<&usize> = sparse_boundary_maps.keys().collect::<Vec<_>>();
    dimensions.sort();
    // dimension hashmap: q: dim of C_q at filtration indices
    for q in dimensions {
        let mut wtr = Writer::from_path(format!("timings_{}.csv", q)).unwrap();
        let global_boundary_map_qm1 = match sparse_boundary_maps.get(&(q - 1)) {
            Some(map) => &map,
            None => &SparseMatrix::from(CooMatrix::zeros(0, 0)),
        };
        let num_qm1_simplices_global = global_boundary_map_qm1.csc.ncols();
        let global_boundary_map_q = match sparse_boundary_maps.get(&q) {
            Some(map) => &map,
            None => &SparseMatrix::from(CooMatrix::zeros(num_qm1_simplices_global, 0)),
        };
        let num_q_simplices_global = global_boundary_map_q.csc.ncols();
        let global_boundary_map_qp1 = match sparse_boundary_maps.get(&(q + 1)) {
            Some(map) => &map,
            None => &SparseMatrix::from(CooMatrix::zeros(num_q_simplices_global, 0)),
        };
        // Initialize results hash of eigenvalues for q-simplices.
        let q_eigenvalues = DashMap::new();

        // First compute the nonzero eigenvalues of the down laplacians
        // TODO: faster to use singular values on the transition matrices!
        let mut down_eigenvalues_q = HashMap::new();
        for k in &filtration_indices {
            let dimension_hashmap_k = filt_hash.get(&k).unwrap();
            let num_q_simplices_k = dimension_hashmap_k.get(&q).unwrap();
            if num_q_simplices_k == &0 {
                // The map is always zero, so we don't care
                continue;
            }
            let num_qm1_simplices_k = dimension_hashmap_k.get(&(q - 1)).unwrap_or(&0);
            if num_qm1_simplices_k > &0 {
                let instant = Instant::now();
                let down = compute_down_persistent_laplacian_transposing(
                    *num_qm1_simplices_k,
                    *num_q_simplices_k,
                    &global_boundary_map_q.csr,
                    zero_tol,
                );
                let elapsed = instant.elapsed().as_secs_f64().to_string();
                wtr.write_record(&[
                    "d",
                    "m",
                    &num_q_simplices_k.to_string(),
                    &num_q_simplices_k.to_string(),
                    &elapsed,
                ])
                .unwrap();
                let instant = Instant::now();
                let down_nonzero_eigs =
                    compute_nonzero_eigenvalues(&down, num_nonzero_eigenvalues, zero_tol);
                let elapsed = instant.elapsed().as_secs_f64().to_string();
                wtr.write_record(&[
                    "d",
                    "e",
                    &num_q_simplices_k.to_string(),
                    &num_q_simplices_k.to_string(),
                    &elapsed,
                ])
                .unwrap();
                down_eigenvalues_q.insert(k, down_nonzero_eigs);
            }
        }

        // Compute up persistent laplacians and their eigenvalues
        for l in &filtration_indices {
            let dimension_hashmap_l = filt_hash.get(l).unwrap();
            let num_q_simplices_l = dimension_hashmap_l.get(&q).unwrap();
            // Only have a boundary map if there are higher dimensional simplices
            let num_qp1_simplices_l = dimension_hashmap_l.get(&(q + 1)).unwrap_or(&0);
            let boundary_map_l_qp1: Option<CsrMatrix<f64>> = if num_qp1_simplices_l > &0 {
                Some(
                    upper_submatrix_csr(
                        &global_boundary_map_qp1.csr,
                        *num_q_simplices_l,
                        *num_qp1_simplices_l,
                        zero_tol,
                    )
                    .into(),
                )
            } else {
                None
            };
            let mut up_persistent_laplacian_option =
                boundary_map_l_qp1.map(|b| up_laplacian_transposing(&b));
            // For each filtration value lower than the current filtration, compute the persistent laplacian.
            let lower_indices = match filtration_indices.iter().position(|&x| x <= *l) {
                Some(idx) => filtration_indices[idx..].to_vec(),
                None => vec![],
            };
            for k in lower_indices {
                let dimension_hashmap_k = filt_hash.get(&k).unwrap();
                let num_q_simplices_k = dimension_hashmap_k.get(&q).unwrap();
                if num_q_simplices_k == &0 {
                    // The map is always zero, so we don't care
                    continue;
                }
                // Compute the up persistent laplacian for K \hookrightarrow L inductively
                let instant = Instant::now();
                up_persistent_laplacian_option = up_persistent_laplacian_option
                    .and_then(|u| compute_up_persistent_laplacian(*num_q_simplices_k, u, zero_tol));
                let elapsed = instant.elapsed().as_secs_f64().to_string();
                wtr.write_record(&[
                    "u",
                    "m",
                    &num_q_simplices_k.to_string(),
                    &num_q_simplices_l.to_string(),
                    &elapsed,
                ])
                .unwrap();
                let instant = Instant::now();
                let up_persistent_nonzero_eigs = match up_persistent_laplacian_option.as_ref() {
                    Some(up_persistent_laplacian) => compute_nonzero_eigenvalues(
                        &up_persistent_laplacian,
                        num_nonzero_eigenvalues,
                        zero_tol,
                    ),
                    None => vec![],
                };
                let elapsed = instant.elapsed().as_secs_f64().to_string();
                wtr.write_record(&[
                    "u",
                    "e",
                    &num_q_simplices_k.to_string(),
                    &num_q_simplices_l.to_string(),
                    &elapsed,
                ])
                .unwrap();
                // Get the smallest across down and up
                let down_eigs = down_eigenvalues_q.get(&k).cloned().unwrap_or_default();
                let mut smallest_nonzero_eigs: Vec<f64> = Vec::new();
                smallest_nonzero_eigs.extend_from_slice(&down_eigs);
                smallest_nonzero_eigs.extend_from_slice(&up_persistent_nonzero_eigs);
                smallest_nonzero_eigs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                smallest_nonzero_eigs.truncate(num_nonzero_eigenvalues);
                q_eigenvalues.insert((k, *l), smallest_nonzero_eigs);
            }
        }
        eigenvalues.insert(*q, q_eigenvalues.into_iter().collect());
    }
    eigenvalues
}

/// Compute persistent homology using persistent Laplacians.
///
/// Args:
///     boundary_maps (dict): A dictionary mapping dimensions to sparse boundary maps.
///     filt (dict): A nested dictionary mapping filtration steps to simplex counts per dimension.
///     zero_tol (float): Tolerance below which values are considered zero.
///
/// Returns:
///     dict: A nested dictionary of persistent homology ranks for each pair.
#[pyfunction]
pub fn process_tda(
    py: Python,
    boundary_maps: &PyDict,
    filt: &PyDict,
    zero_tol: f64,
) -> PyResult<PyObject> {
    let sparse_boundary_maps = process_sparse_dict(boundary_maps).unwrap();

    let filt_hash = parse_nested_dict(&filt).unwrap();
    let eigenvalues = persistent_homology_of_filtration(
        sparse_boundary_maps,
        filt_hash,
        compute_homology_from_persistent_laplacian_dense,
        zero_tol,
    );
    Ok(eigenvalues.into_py(py))
}

/// Computes the smallest nonzero persistent eigenvalue of persistent Laplacians of a filtration.
///
/// Args:
///     boundary_maps (dict): A dictionary mapping dimensions to sparse boundary maps.
///     filt (dict): A nested dictionary mapping filtration steps to simplex counts per dimension.
///     zero_tol (float): Tolerance below which values are considered zero.
///     filtration_subsampling (list, optional): Indices of the filtration to use when downsampling.
///     use_scipy (bool): If true, uses SciPy's eigsh for eigenvalue computation. Otherwise uses dense computation.
///     use_stepwise_schur (bool): If true, uses stepwise Schur decomposition for up persistent Laplacian computation.
/// Returns:
///     dict: A nested dictionary of the smallest nonzero persistent eigenvalue at each filtration pair.
#[pyfunction]
#[pyo3(signature = (boundary_maps, filt, zero_tol, filtration_subsampling=None, use_scipy=false, use_stepwise_schur=false))]
pub fn smallest_eigenvalue(
    py: Python,
    boundary_maps: &PyDict,
    filt: &PyDict,
    zero_tol: f64,
    filtration_subsampling: Option<Vec<usize>>,
    use_scipy: bool,
    use_stepwise_schur: bool,
) -> PyResult<PyObject> {
    let sparse_boundary_maps = process_sparse_dict(boundary_maps).unwrap();
    let laplacian_compute = if use_stepwise_schur {
        compute_up_persistent_laplacian_stepwise
    } else {
        compute_up_persistent_laplacian_schur
    };

    let filt_hash = parse_nested_dict(&filt).unwrap();
    let eigenvalues = if use_scipy {
        Python::with_gil(|py| {
            let scipy_config =
                ScipyEigshConfig::new_from_num_nonzero_eigenvalues_tol(1, zero_tol, py);
            persistent_eigenvalues_of_filtration(
                sparse_boundary_maps,
                filt_hash,
                laplacian_compute,
                |matrix, _num_nonzero, _zero_tol| {
                    compute_nonzero_eigenvalues_from_persistent_laplacian_scipy(
                        matrix,
                        &scipy_config,
                    )
                },
                1,
                filtration_subsampling,
                zero_tol,
            )
        })
    } else {
        persistent_eigenvalues_of_filtration(
            sparse_boundary_maps,
            filt_hash,
            laplacian_compute,
            compute_nonzero_eigenvalues_from_persistent_laplacian_dense,
            1,
            filtration_subsampling,
            zero_tol,
        )
    };
    Ok(eigenvalues.into_py(py))
}

#[pymodule]
fn persistent_laplacians(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_tda, m)?)?;
    m.add_function(wrap_pyfunction!(smallest_eigenvalue, m)?)?;
    Ok(())
}
