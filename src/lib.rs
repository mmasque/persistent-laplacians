use dashmap::DashMap;
use homology::compute_homology_from_persistent_laplacian_dense;
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use sprs::DenseVector;
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::eigenvalues::{
    compute_eigenvalues_from_persistent_laplacian_lanczos_crate,
    compute_eigenvalues_from_persistent_laplacian_primme_crate,
    compute_nonzero_eigenvalues_from_persistent_laplacian_dense,
};
use crate::homology::{eigsh_scipy, ScipyEigshConfig};
pub mod eigenvalues;
pub mod homology;

pub static TOL: f64 = 1e-6;

// Temporary before I find a better solution.
fn is_float_zero(float: f64) -> bool {
    float.abs() < TOL
}

// Sometimes you want them all
#[derive(Debug, Clone)]
pub struct SparseMatrix<T> {
    pub csc: CscMatrix<T>,
    pub csr: CsrMatrix<T>,
    coo: CooMatrix<T>,
}

impl From<CscMatrix<f64>> for SparseMatrix<f64> {
    fn from(value: CscMatrix<f64>) -> Self {
        let csr = CsrMatrix::from(&value);
        let coo = CooMatrix::from(&value);
        SparseMatrix {
            csc: value,
            csr,
            coo,
        }
    }
}

impl From<CsrMatrix<f64>> for SparseMatrix<f64> {
    fn from(value: CsrMatrix<f64>) -> Self {
        let csc = CscMatrix::from(&value);
        let coo = CooMatrix::from(&value);
        SparseMatrix {
            csc,
            csr: value,
            coo,
        }
    }
}

impl From<CooMatrix<f64>> for SparseMatrix<f64> {
    fn from(value: CooMatrix<f64>) -> Self {
        let csc: CscMatrix<f64> = CscMatrix::from(&value);
        let csr = CsrMatrix::from(&value);
        SparseMatrix {
            csc,
            csr,
            coo: value,
        }
    }
}

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

fn up_degree(boundary_map_qp1: &SparseMatrix<f64>) -> SparseMatrix<f64> {
    // start with a zero matrix. Dimensions of codomain in boundary map out of q+1 simplices
    let csr = &boundary_map_qp1.csr;
    let domain_dimension = csr.nrows();

    // Degree computation
    // for each q-simplex i (think vertex), look at each q+1 simplex j (think edge) containing it
    // j contains i iff boundary_map_qp1(i, j) != 0
    // So the number of nonzero entries in the ith row is what we want. This is easy using row_offsets
    let row_offsets = csr.row_offsets();
    let mut degrees = Vec::with_capacity(domain_dimension);
    let mut rows = Vec::with_capacity(domain_dimension);
    let mut cols = Vec::with_capacity(domain_dimension);
    for i in 0..domain_dimension {
        degrees.push((row_offsets[i + 1] - row_offsets[i]) as f64);
        rows.push(i);
        cols.push(i);
    }
    let coo = CooMatrix::try_from_triplets(domain_dimension, domain_dimension, rows, cols, degrees)
        .unwrap();
    return SparseMatrix::from(coo);
}

fn up_adjacency(boundary_map_qp1: &SparseMatrix<f64>) -> SparseMatrix<f64> {
    // For each (q+1) simplex l look at all its q simplices
    // For each pair (i, j) of q simplices of the q+1 simplex
    // boundary_map_qp1(i, l) says if l contains i, with +-1.
    // boundary_map_qp1(j, l) says if l contains j, with +-1.

    // boundary map data
    let csc = &boundary_map_qp1.csc;

    // new matrix data
    let mut new_rows = vec![];
    let mut new_cols = vec![];
    let mut new_data = vec![];

    for col in csc.col_iter() {
        let col_rows = col.row_indices();
        let col_values = col.values();
        for i_ind in 0..col_rows.len() {
            // The global row index i
            for j_ind in (i_ind + 1)..col_rows.len() {
                let val = -col_values[i_ind] * col_values[j_ind];
                new_rows.push(col_rows[i_ind]);
                new_cols.push(col_rows[j_ind]);
                new_data.push(val);

                new_rows.push(col_rows[j_ind]);
                new_cols.push(col_rows[i_ind]);
                new_data.push(val);
            }
        }
    }

    let nrows = csc.nrows();
    let coo = CooMatrix::try_from_triplets(nrows, nrows, new_rows, new_cols, new_data).unwrap();
    SparseMatrix::from(coo)
}

fn down_degree(boundary_map_q: &SparseMatrix<f64>) -> SparseMatrix<f64> {
    let csc = &boundary_map_q.csc;
    let domain_dim = csc.ncols();
    let mut degrees: Vec<f64> = Vec::zeros(domain_dim);
    let rows: Vec<usize> = (0..domain_dim).collect();
    let cols: Vec<usize> = (0..domain_dim).collect();

    for (l, col) in csc.col_iter().enumerate() {
        let row_indices = col.row_indices();
        degrees[l] += row_indices.len() as f64;
    }
    let coo = CooMatrix::try_from_triplets(domain_dim, domain_dim, rows, cols, degrees).unwrap();
    SparseMatrix::from(coo)
}

fn down_adjacency(boundary_map_q: &SparseMatrix<f64>) -> SparseMatrix<f64> {
    // For each pair of q simplices (i, j), if they have no common q-1 simplex (face)
    // then A(i, j) = 0. If they have a common q-1 simplex l, then A(i, j) = - B(i, l) * B(j, l).
    // q simplices -> q-1 simplices, so q simplices are columns, q-1 simplices are rows
    let csc = &boundary_map_q.csc;

    let mut new_data = vec![];
    let mut new_cols = vec![];
    let mut new_rows = vec![];

    let ncols = csc.ncols();
    for i in 0..ncols {
        let ith_col = csc.col(i);
        let ith_values = ith_col.values();
        let ith_rows = ith_col.row_indices();
        for j in (i + 1)..ncols {
            let jth_col = csc.col(j);
            let jth_values = jth_col.values();
            let jth_rows = jth_col.row_indices();
            // Two pointer intersection
            let (mut p, mut q) = (0, 0);
            while p < ith_col.nnz() && q < jth_col.nnz() {
                match ith_rows[p].cmp(&jth_rows[q]) {
                    Ordering::Equal => break,
                    Ordering::Less => p += 1,
                    Ordering::Greater => q += 1,
                }
            }
            if p < ith_col.nnz() && q < jth_col.nnz() {
                let prod = -ith_values[p] * jth_values[q];
                new_data.push(prod);
                new_cols.push(j);
                new_rows.push(i);

                new_data.push(prod);
                new_cols.push(i);
                new_rows.push(j);
            }
        }
    }
    let domain_dim = csc.ncols();
    let coo =
        CooMatrix::try_from_triplets(domain_dim, domain_dim, new_rows, new_cols, new_data).unwrap();
    SparseMatrix::from(coo)
}

fn up_laplacian(boundary_map_qp1: &SparseMatrix<f64>) -> SparseMatrix<f64> {
    let degree_matrix = up_degree(boundary_map_qp1);
    let adjacency_matrix = up_adjacency(boundary_map_qp1);
    (degree_matrix.csr - adjacency_matrix.csr).into()
}

pub fn up_laplacian_transposing(boundary_map_qp1: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    boundary_map_qp1 * boundary_map_qp1.transpose()
}

fn down_laplacian(boundary_map_q: &SparseMatrix<f64>) -> SparseMatrix<f64> {
    let degree_matrix = down_degree(boundary_map_q);
    let adjacency_matrix = down_adjacency(boundary_map_q);
    (degree_matrix.csr - adjacency_matrix.csr).into()
}

fn down_laplacian_transposing(boundary_map_q: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    boundary_map_q.transpose() * boundary_map_q
}

// Implementation of Theorem 5.1 in Memoli, equation 15.
fn up_persistent_laplacian_step(
    prev_up_persistent_laplacian: CsrMatrix<f64>,
) -> Option<CsrMatrix<f64>> {
    // If the bottom right entry is zero in our input matrix, then return the previous laplacian
    // without the last row and column.
    let dim_prev = prev_up_persistent_laplacian.ncols();
    let new_dim = dim_prev.checked_sub(1)?;
    // Note: if check incurs binary search cost, so log(dim_prev). But the computation inside is already linear.
    let bottom_right = prev_up_persistent_laplacian.get_entry(new_dim, new_dim);
    let (bottom_right_is_zero, bottom_right_value) = bottom_right
        .map(|x| {
            let value = x.into_value();
            (is_float_zero(value), value)
        })
        .unwrap_or((true, 0.0));
    let exclude_last_row_col = drop_last_row_col_csr(&prev_up_persistent_laplacian);
    if !bottom_right_is_zero {
        // prev(i, j) - prev(i, dim_prev) * prev(dim_prev, j) / prev(dim_prev, dim_prev)
        let outer =
            outer_product_last_col_row_csr(&prev_up_persistent_laplacian) / bottom_right_value;
        return Some(exclude_last_row_col - outer);
    } else {
        return Some(exclude_last_row_col);
    }
}

/// Computes the qth up persistent laplacian of a pair of simplicial complexes K hookrightarrow L given the qth up laplacian
/// of L and the number of q simplices of K.
pub fn compute_up_persistent_laplacian(
    num_q_simplices_k: usize,
    up_laplacian: CsrMatrix<f64>,
) -> CsrMatrix<f64> {
    assert!(num_q_simplices_k > 0);
    let lower_by = up_laplacian.ncols() - num_q_simplices_k;
    let mut new_up_persistent_laplacian = up_laplacian;
    // We can only lower by 1 at a time, so if lower_by > 1, we take it step by step
    for _ in 1..=lower_by {
        new_up_persistent_laplacian =
            up_persistent_laplacian_step(new_up_persistent_laplacian).unwrap();
    }
    new_up_persistent_laplacian
}

fn compute_down_persistent_laplacian(
    num_qm1_simplices_k: usize,
    num_q_simplices_k: usize,
    global_boundary_map_q: &SparseMatrix<f64>,
) -> SparseMatrix<f64> {
    let boundary_map_q_k: CsrMatrix<f64> = CsrMatrix::from(&upper_submatrix(
        &global_boundary_map_q.coo,
        num_qm1_simplices_k,
        num_q_simplices_k,
    ));
    let down_persistent_laplacian = down_laplacian(&SparseMatrix::from(boundary_map_q_k));
    down_persistent_laplacian
}

pub fn compute_down_persistent_laplacian_transposing(
    num_qm1_simplices_k: usize,
    num_q_simplices_k: usize,
    global_boundary_map_q: &CsrMatrix<f64>,
) -> CsrMatrix<f64> {
    let q_boundary_k = upper_submatrix_csr(
        &global_boundary_map_q,
        num_qm1_simplices_k,
        num_q_simplices_k,
    );
    let down_persistent_laplacian = down_laplacian_transposing(&q_boundary_k);
    down_persistent_laplacian
}

fn to_dense(csr: &CsrMatrix<f64>) -> DMatrix<f64> {
    let nrows = csr.nrows();
    let ncols = csr.ncols();
    let mut dense = DMatrix::<f64>::zeros(nrows, ncols);

    // iterate all stored (i, j, v) triplets and write into the dense matrix
    for (i, j, v) in csr.triplet_iter() {
        dense[(i, j)] = *v;
    }
    dense
}

/// Computes persistent homology of a filtration using persistent laplacians
/// Assumes number of (q) simplices increases by at most 1 on each step of filtration
pub fn persistent_homology_of_filtration<F>(
    sparse_boundary_maps: HashMap<usize, SparseMatrix<f64>>,
    // filtration_index: {q: dimension_q_simplices}
    filt_hash: HashMap<usize, HashMap<usize, usize>>,
    compute_homology_from_persistent_laplacian: F,
) -> HashMap<usize, HashMap<(usize, usize), usize>>
where
    F: Fn(&CsrMatrix<f64>) -> usize,
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
        // For performance evaluation of python's scipy, cannot have par iters, because
        // Python object from Pyo3 is not sync.
        // filtration_indices
        //     .par_iter()
        //     .filter(|l| {
        //         let dimension_hashmap_l = filt_hash.get(l).unwrap();
        //         let num_q_simplices_l = dimension_hashmap_l.get(&q).unwrap();
        //         num_q_simplices_l != &0
        //     })
        //     .for_each(|l| {
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
                up_persistent_laplacian = up_persistent_laplacian
                    .map(|u| compute_up_persistent_laplacian(*num_q_simplices_k, u));
                // Compute the down persistent laplacian for K \hookrightarrow L
                // If there are no lower simplices, the map factors via the 0 vector space, so it is zero
                let down_persistent_laplacian = if num_qm1_simplices_k > &0 {
                    Some(compute_down_persistent_laplacian_transposing(
                        *num_qm1_simplices_k,
                        *num_q_simplices_k,
                        &global_boundary_map_q.csr,
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
                        compute_homology_from_persistent_laplacian(&persistent_laplacian);

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

/// Computes some nonzero persistent eigenvalues of a filtration
/// Assumes number of (q) simplices increases by at most 1 on each step of filtration
pub fn persistent_eigenvalues_of_filtration<F>(
    sparse_boundary_maps: HashMap<usize, SparseMatrix<f64>>,
    // filtration_index: {q: dimension_q_simplices}
    filt_hash: HashMap<usize, HashMap<usize, usize>>,
    compute_nonzero_eigenvalues: F,
    num_nonzero_eigenvalues: usize,
    // Indices of the filtration to use when downsampling
    downsampled_filtration_indices: Option<Vec<usize>>,
) -> HashMap<usize, HashMap<(usize, usize), Vec<f64>>>
where
    F: Fn(&CsrMatrix<f64>, usize) -> Vec<f64>,
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
                let down = compute_down_persistent_laplacian_transposing(
                    *num_qm1_simplices_k,
                    *num_q_simplices_k,
                    &global_boundary_map_q.csr,
                );
                let down_nonzero_eigs = compute_nonzero_eigenvalues(&down, num_nonzero_eigenvalues);
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
                up_persistent_laplacian_option = up_persistent_laplacian_option
                    .map(|u| compute_up_persistent_laplacian(*num_q_simplices_k, u));
                let up_persistent_nonzero_eigs = match up_persistent_laplacian_option.as_ref() {
                    Some(up_persistent_laplacian) => compute_nonzero_eigenvalues(
                        &up_persistent_laplacian,
                        num_nonzero_eigenvalues,
                    ),
                    None => vec![],
                };
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

#[pyfunction]
fn process_tda(py: Python, boundary_maps: &PyDict, filt: &PyDict) -> PyResult<PyObject> {
    let sparse_boundary_maps = process_sparse_dict(boundary_maps).unwrap();

    let filt_hash = parse_nested_dict(&filt).unwrap();
    let eigenvalues = persistent_homology_of_filtration(
        sparse_boundary_maps,
        filt_hash,
        compute_homology_from_persistent_laplacian_dense,
    );
    Ok(eigenvalues.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (boundary_maps, filt, filtration_subsampling=None, use_scipy=false))]
fn smallest_eigenvalue(
    py: Python,
    boundary_maps: &PyDict,
    filt: &PyDict,
    filtration_subsampling: Option<Vec<usize>>,
    use_scipy: bool,
) -> PyResult<PyObject> {
    let sparse_boundary_maps = process_sparse_dict(boundary_maps).unwrap();

    let filt_hash = parse_nested_dict(&filt).unwrap();
    let eigenvalues = if use_scipy {
        Python::with_gil(|py| {
            let scipy_config = ScipyEigshConfig::default_from_num_nonzero_eigenvalues(1, py);
            persistent_eigenvalues_of_filtration(
                sparse_boundary_maps,
                filt_hash,
                |matrix, _num_nonzero| eigsh_scipy(matrix, &scipy_config).unwrap_or(vec![]),
                1,
                filtration_subsampling,
            )
        })
    } else {
        persistent_eigenvalues_of_filtration(
            sparse_boundary_maps,
            filt_hash,
            compute_eigenvalues_from_persistent_laplacian_primme_crate,
            1,
            filtration_subsampling,
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

/// Compute the product C[:n]R[:n] of the last column C by the last row R of an n x n matrix,
/// without the last entry, so returning an (n-1) x (n-1) matrix
fn outer_product_last_col_row(sparse: &SparseMatrix<f64>) -> CooMatrix<f64> {
    let csc = &sparse.csc;
    let csr = &sparse.csr;
    let n = csc.ncols();
    assert_eq!(n, csc.nrows());
    assert_eq!(n, csr.ncols());
    assert_eq!(n, csr.nrows());

    let col = csc.col(n - 1);
    let row = csr.row(n - 1);

    let mut coo = CooMatrix::new(n - 1, n - 1);

    for (&i, &v_col) in col.row_indices().iter().zip(col.values()) {
        if i == n - 1 {
            continue;
        }
        for (&j, &v_row) in row.col_indices().iter().zip(row.values()) {
            if j == n - 1 {
                continue;
            }
            coo.push(i, j, v_col * v_row);
        }
    }
    coo
}

fn outer_product_last_col_row_csr(csr: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let n = csr.nrows();
    assert_eq!(n, csr.ncols());
    let last = n - 1;
    let indptr = csr.row_offsets();
    let col_indices = csr.col_indices();
    let values = csr.values();

    // 1) Extract row = A[last, :]
    let row_start = indptr[last];
    let row_end = indptr[last + 1];
    let row_idxs = &col_indices[row_start..row_end];
    let row_vals = &values[row_start..row_end];

    // 2) Build a length-n vector of the last column entries v[i] = A[i, last]
    let mut col_vec = vec![0.0; n];
    for i in 0..n {
        let start = indptr[i];
        let end = indptr[i + 1];
        // scan that row’s indices for column = last
        for idx in start..end {
            if col_indices[idx] == last {
                col_vec[i] = values[idx];
                break;
            }
        }
    }

    // 3) Now build the outer product of col_vec[0..n-1] and row_vals[ j < n-1 ]
    let m = n - 1;
    // precompute row-nnz and total nnz
    let row_nnz = row_idxs.iter().take_while(|&&j| j < m).count();
    // each i with col_vec[i] != 0 contributes row_nnz entries
    let mut indptr_new = Vec::with_capacity(m + 1);
    indptr_new.push(0);
    let mut offset = 0;
    for i in 0..m {
        if col_vec[i] != 0.0 {
            offset += row_nnz;
        }
        indptr_new.push(offset);
    }

    let mut indices = Vec::with_capacity(offset);
    let mut data = Vec::with_capacity(offset);
    for i in 0..m {
        let v = col_vec[i];
        if v != 0.0 {
            for (&j, &w) in row_idxs.iter().zip(row_vals.iter()) {
                if j < m {
                    indices.push(j);
                    data.push(v * w);
                }
            }
        }
    }
    CsrMatrix::try_from_csr_data(m, m, indptr_new, indices, data).unwrap()
}

fn drop_last_row_col_coo(matrix: &CooMatrix<f64>) -> CooMatrix<f64> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    upper_submatrix(matrix, nrows - 1, ncols - 1)
}

fn drop_last_row_col_csr(matrix: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    upper_submatrix_csr(matrix, nrows - 1, ncols - 1)
}

/// Take upper left submatrix
/// rows and cols are the dimensions of the submatrix to take
/// TODO: maybe take ownership, then early return when rows = nrows, cols = ncols
fn upper_submatrix(matrix: &CooMatrix<f64>, rows: usize, cols: usize) -> CooMatrix<f64> {
    // Temporary asserts to avoid annoying size bugs
    assert!(rows > 0);
    assert!(cols > 0);
    let mut new_coo = CooMatrix::new(rows, cols);
    for (i, j, v) in matrix.triplet_iter() {
        if i < rows && j < cols {
            new_coo.push(i, j, *v);
        }
    }
    new_coo
}

/// Removes extra zeros.
pub fn upper_submatrix_csr(matrix: &CsrMatrix<f64>, rows: usize, cols: usize) -> CsrMatrix<f64> {
    assert!(rows > 0 && cols > 0);
    let indptr = matrix.row_offsets();
    let col_idx = matrix.col_indices();
    let vals = matrix.values();

    // 1) Build new indptr for rows [0..rows)
    let mut new_indptr = Vec::with_capacity(rows + 1);
    new_indptr.push(0);
    let mut nnz_acc = 0;
    for r in 0..rows {
        let start = indptr[r];
        let end = indptr[r + 1];
        let mut cnt = 0;
        // Count entries in this row within the column limit and above threshold
        for idx in start..end {
            let j = col_idx[idx];
            let v = vals[idx];
            if j < cols && !is_float_zero(v) {
                cnt += 1;
            }
        }
        nnz_acc += cnt;
        new_indptr.push(nnz_acc);
    }

    // 2) Allocate storage for indices & data
    let mut new_cols = Vec::with_capacity(nnz_acc);
    let mut new_vals = Vec::with_capacity(nnz_acc);

    // 3) Fill them by a second pass
    for r in 0..rows {
        let start = indptr[r];
        let end = indptr[r + 1];
        for idx in start..end {
            let j = col_idx[idx];
            let v = vals[idx];
            if j < cols && !is_float_zero(v) {
                new_cols.push(j);
                new_vals.push(v);
            }
        }
    }

    // 4) Build and return the new CSR matrix
    CsrMatrix::try_from_csr_data(rows, cols, new_indptr, new_cols, new_vals)
        .expect("submatrix dimensions and nnz must be consistent")
}
pub fn count_zeros(data: DVector<f64>, eps: f64) -> usize {
    data.iter().filter(|&&x| x.abs() <= eps).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
    use pyo3::types::PyList;
    #[test]
    fn test_up_laplacian_triangle() {
        // Boundary map d1 (edges -> vertices)
        // 3 vertices (rows), 2 edges (columns)
        // Edge 0: Vertex 0 -> 1
        // Edge 1: Vertex 1 -> 2
        // Edge 2: Vertex 0 -> 2
        // 01 12 02
        // 0       -1   0. -1
        // 1       1. -1.  0
        // 2      0.   1.  1
        let mut coo_boundary = CooMatrix::new(3, 3);
        coo_boundary.push(0, 0, -1.0);
        coo_boundary.push(0, 2, -1.0);
        coo_boundary.push(1, 0, 1.0);
        coo_boundary.push(1, 1, -1.0);
        coo_boundary.push(2, 1, 1.0);
        coo_boundary.push(2, 2, 1.0);
        let boundary_map = SparseMatrix::from(coo_boundary);
        let lap = up_laplacian(&boundary_map);

        // Expected up Laplacian (degree - adjacency):
        // [ 2  -1  -1 ]
        // [-1  2  -1 ]
        // [ -1  -1  2 ]
        let mut expected_data_coo = CooMatrix::new(3, 3);
        expected_data_coo.push(0, 0, 2.0);
        expected_data_coo.push(0, 1, -1.0);
        expected_data_coo.push(0, 2, -1.0);
        expected_data_coo.push(1, 0, -1.0);
        expected_data_coo.push(1, 1, 2.0);
        expected_data_coo.push(1, 2, -1.0);
        expected_data_coo.push(2, 0, -1.0);
        expected_data_coo.push(2, 1, -1.0);
        expected_data_coo.push(2, 2, 2.0);
        assert_eq!(expected_data_coo, lap.coo)
    }

    #[test]
    fn test_up_laplacian_2simplex() {
        let mut coo_boundary = CooMatrix::new(3, 1);
        coo_boundary.push(0, 0, 1.0);
        coo_boundary.push(1, 0, 1.0);
        coo_boundary.push(2, 0, -1.0);
        let boundary_map = SparseMatrix::from(coo_boundary);
        let lap = up_laplacian(&boundary_map);

        // Expected up Laplacian (degree - adjacency):
        // 1 1 -1
        // 1 1 -1
        // -1 -1 1
        let mut expected_data_coo = CooMatrix::new(3, 3);
        expected_data_coo.push(0, 0, 1.0);
        expected_data_coo.push(0, 1, 1.0);
        expected_data_coo.push(0, 2, -1.0);
        expected_data_coo.push(1, 0, 1.0);
        expected_data_coo.push(1, 1, 1.0);
        expected_data_coo.push(1, 2, -1.0);
        expected_data_coo.push(2, 0, -1.0);
        expected_data_coo.push(2, 1, -1.0);
        expected_data_coo.push(2, 2, 1.0);
        assert_eq!(expected_data_coo, lap.coo)
    }

    #[test]
    fn test_down_laplacian_2simplex() {
        let mut coo_boundary = CooMatrix::new(3, 1);
        coo_boundary.push(0, 0, 1.0);
        coo_boundary.push(1, 0, 1.0);
        coo_boundary.push(2, 0, -1.0);
        let boundary_map = SparseMatrix::from(coo_boundary);
        let lap = down_laplacian(&boundary_map);

        // Expected down Laplacian (degree - adjacency):
        let mut expected_data_coo = CooMatrix::new(1, 1);
        expected_data_coo.push(0, 0, 3.0);
        assert_eq!(expected_data_coo, lap.coo)
    }

    #[test]
    fn test_down_laplacian_triangle() {
        // Boundary map d1 (edges -> vertices)
        // 3 vertices (rows), 2 edges (columns)
        // Edge 0: Vertex 0 -> 1
        // Edge 1: Vertex 1 -> 2
        // Edge 2: Vertex 0 -> 2
        // 01 12 02
        // 0       -1   0. -1
        // 1       1. -1.  0
        // 2      0.   1.  1
        let mut coo_boundary = CooMatrix::new(3, 3);
        coo_boundary.push(0, 0, -1.0);
        coo_boundary.push(0, 2, -1.0);
        coo_boundary.push(1, 0, 1.0);
        coo_boundary.push(1, 1, -1.0);
        coo_boundary.push(2, 1, 1.0);
        coo_boundary.push(2, 2, 1.0);
        let boundary_map = SparseMatrix::from(coo_boundary);
        let lap = down_laplacian(&boundary_map);

        // Expected down Laplacian (degree - adjacency):
        // [ 2  -1  1 ]
        // [-1  2  1 ]
        // [ 1  1  2 ]
        let mut expected_data_coo = CooMatrix::new(3, 3);
        expected_data_coo.push(0, 0, 2.0);
        expected_data_coo.push(0, 1, -1.0);
        expected_data_coo.push(0, 2, 1.0);
        expected_data_coo.push(1, 0, -1.0);
        expected_data_coo.push(1, 1, 2.0);
        expected_data_coo.push(1, 2, 1.0);
        expected_data_coo.push(2, 0, 1.0);
        expected_data_coo.push(2, 1, 1.0);
        expected_data_coo.push(2, 2, 2.0);
        assert_eq!(expected_data_coo, lap.coo)
    }

    #[test]
    fn test_up_persistent_laplacian_step_triangle() {
        let mut coo_boundary = CooMatrix::new(3, 3);
        coo_boundary.push(0, 0, -1.0);
        coo_boundary.push(0, 2, -1.0);
        coo_boundary.push(1, 0, 1.0);
        coo_boundary.push(1, 1, -1.0);
        coo_boundary.push(2, 1, 1.0);
        coo_boundary.push(2, 2, 1.0);
        let boundary_map = CsrMatrix::from(&coo_boundary);
        // Boundary map d1 (edges -> vertices)
        // 3 vertices (rows), 2 edges (columns)
        // Edge 0: Vertex 0 -> 1
        // Edge 1: Vertex 1 -> 2
        // Edge 2: Vertex 0 -> 2
        // 01 12 02
        // 0       -1   0. -1
        // 1       1. -1.  0
        // 2      0.   1.  1
        // persistence in the last step in an imagined filtration, where there is one fewer 1-simplex (edge)
        let up_laplacian = up_laplacian_transposing(&boundary_map);
        let persistent_up = up_persistent_laplacian_step(up_laplacian.into()).unwrap();
        let mut expected = CooMatrix::new(2, 2);
        expected.push(0, 0, 1.5);
        expected.push(0, 1, -1.5);
        expected.push(1, 0, -1.5);
        expected.push(1, 1, 1.5);
        assert_eq!(expected, CooMatrix::from(&persistent_up))
    }

    // https://arxiv.org/abs/2312.07563 Example 3.4
    #[test]
    fn test_up_persistent_laplacian_step_example_3_4() {
        let mut coo_boundary = CooMatrix::new(5, 2);
        coo_boundary.push(0, 0, 1.0);
        coo_boundary.push(1, 0, 1.0);
        coo_boundary.push(4, 0, -1.0);

        coo_boundary.push(2, 1, 1.0);
        coo_boundary.push(3, 1, -1.0);
        coo_boundary.push(4, 1, 1.0);
        let up_laplacian = up_laplacian_transposing(&CsrMatrix::from(&coo_boundary));
        let up_persistent_laplacian = up_persistent_laplacian_step(up_laplacian).unwrap();

        let mut expected = CooMatrix::new(4, 4);
        expected.push(0, 0, 1.0);
        expected.push(0, 1, 1.0);
        expected.push(0, 2, 1.0);
        expected.push(0, 3, -1.0);

        expected.push(1, 0, 1.0);
        expected.push(1, 1, 1.0);
        expected.push(1, 2, 1.0);
        expected.push(1, 3, -1.0);

        expected.push(2, 0, 1.0);
        expected.push(2, 1, 1.0);
        expected.push(2, 2, 1.0);
        expected.push(2, 3, -1.0);

        expected.push(3, 0, -1.0);
        expected.push(3, 1, -1.0);
        expected.push(3, 2, -1.0);
        expected.push(3, 3, 1.0);

        assert_eq!(0.5 * CsrMatrix::from(&expected), up_persistent_laplacian);
    }

    fn test_python_lanczos() {
        // Initialize Python interpreter
        Python::with_gil(|py| {
            // Build a small example CSRMatrix in Rust:
            // A = [[2, -1, 0],
            //      [-1, 2, -1],
            //      [0, -1, 2]]
            let sys = py.import("sys").unwrap();
            let path: &PyList = sys.getattr("path").unwrap().downcast().unwrap();
            path.insert(0, ".venv/lib/python3.10/site-packages")
                .unwrap();

            let mut builder = CooMatrix::zeros(10, 10);
            builder.push(0, 0, 2.0);
            builder.push(0, 1, -1.0);
            builder.push(1, 0, -1.0);
            builder.push(1, 1, 2.0);
            builder.push(1, 2, -1.0);
            builder.push(2, 1, -1.0);
            builder.push(2, 2, 2.0);

            let a = CsrMatrix::from(&builder);
            let eigsh = PyModule::import(py, "scipy.sparse.linalg")
                .unwrap()
                .getattr("eigsh")
                .unwrap();

            let scipy_sparse = PyModule::import(py, "scipy.sparse").unwrap();
            //     let eigs = eigsh_scipy(
            //         py,
            //         &a,
            //         2,
            //         Some(0.00001),
            //         None,
            //         None,
            //         "LM",
            //         eigsh,
            //         scipy_sparse,
            //     )
            //     .unwrap();
            //     println!("Smallest eigenvalues: {:?}", eigs);
        });
    }
}
