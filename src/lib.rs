use core::num;
use lanczos::{Hermitian, HermitianEigen, Order};
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sprs::DenseVector;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::panic::catch_unwind;
use std::ptr::null;

// Temporary before I find a better solution.
fn is_float_zero(float: f64) -> bool {
    float < 1e-4
}

// Sometimes you want them all
#[derive(Debug)]
pub struct SparseMatrix<T> {
    csc: CscMatrix<T>,
    csr: CsrMatrix<T>,
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

fn parse_nested_dict(filt: &PyDict) -> PyResult<HashMap<usize, HashMap<usize, usize>>> {
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

fn process_sparse_dict(dict: &PyDict) -> PyResult<HashMap<usize, SparseMatrix<f64>>> {
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

fn down_laplacian(boundary_map_q: &SparseMatrix<f64>) -> SparseMatrix<f64> {
    let degree_matrix = down_degree(boundary_map_q);
    let adjacency_matrix = down_adjacency(boundary_map_q);
    (degree_matrix.csr - adjacency_matrix.csr).into()
}

// Implementation of Theorem 5.1 in Memoli, equation 15.
fn up_persistent_laplacian_step(
    prev_up_persistent_laplacian: SparseMatrix<f64>,
) -> Option<SparseMatrix<f64>> {
    // If the bottom right entry is zero in our input matrix, then return the previous laplacian
    // without the last row and column.
    let csc = &prev_up_persistent_laplacian.csc;
    let coo = &prev_up_persistent_laplacian.coo;
    let dim_prev = csc.ncols();
    let new_dim = dim_prev.checked_sub(1)?;
    // Note: if check incurs binary search cost, so log(dim_prev). But the computation inside is already linear.
    let bottom_right = csc.get_entry(new_dim, new_dim);
    let (bottom_right_is_zero, bottom_right_value) = bottom_right
        .map(|x| {
            let value = x.into_value();
            (is_float_zero(value), value)
        })
        .unwrap_or((true, 0.0));
    if !bottom_right_is_zero {
        // prev(i, j) - prev(i, dim_prev) * prev(dim_prev, j) / prev(dim_prev, dim_prev)
        let outer_coo = outer_product_last_col_row(&prev_up_persistent_laplacian);
        let outer_weighed = CsrMatrix::from(&outer_coo) / bottom_right_value;
        let exclude_last_row_col_coo = drop_last_row_col_coo(&coo);
        let exclude_last_row_col = CsrMatrix::from(&exclude_last_row_col_coo);

        return Some(SparseMatrix::from(exclude_last_row_col - outer_weighed));
    } else {
        let mut coo = CooMatrix::new(new_dim, new_dim);
        let num_cols_to_edit = new_dim.checked_sub(1).unwrap_or(0);
        for col in 0..num_cols_to_edit {
            let col_view = csc.col_iter().nth(col).unwrap();
            let rows = col_view.row_indices();
            let vals = col_view.values();

            for (&r, &v) in rows.iter().zip(vals.iter()) {
                if r < num_cols_to_edit {
                    coo.push(r, col, v);
                }
            }
        }
        Some(SparseMatrix::from(coo))
    }
}

// Implementation of Theorem 5.1 algorithm in Memoli: https://arxiv.org/pdf/2012.02808
pub fn up_persistent_laplacian(
    boundary_map_qp1: &SparseMatrix<f64>,
    lower_dim_by: usize,
) -> Option<SparseMatrix<f64>> {
    let mut current_laplacian = up_laplacian(&boundary_map_qp1);
    for _ in 0..lower_dim_by {
        let next_laplacian = up_persistent_laplacian_step(current_laplacian)?;
        current_laplacian = next_laplacian;
    }
    return Some(current_laplacian);
}

// Given the global boundary maps d_{q+1}: (q+1) -> q and d_q: q -> (q-1), need
// the dimensions of K for d_{q+1} and d_q,
// the dimensions of L for d_{q+1} and d_q.
pub fn persistent_laplacian(
    global_boundary_map_qp1: &SparseMatrix<f64>,
    dims_qp1_smaller: (usize, usize),
    dims_qp1_larger: (usize, usize),
    global_boundary_map_q: &SparseMatrix<f64>,
    dims_q_smaller: (usize, usize),
) -> Option<SparseMatrix<f64>> {
    let (rows_q_smaller, cols_q_smaller) = dims_q_smaller;
    // Quite slow, since we are throwing away the csc and csr and remaking them
    let boundary_map_q_smaller =
        upper_submatrix(&global_boundary_map_q.coo, rows_q_smaller, cols_q_smaller).into();
    let down_persistent_laplacian = down_laplacian(&boundary_map_q_smaller);
    let (rows_qp1_larger, cols_qp1_larger) = dims_qp1_larger;
    let (rows_qp1_smaller, _) = dims_qp1_smaller; // TODO: cols don't matter?
    let boundary_map_qp1_larger: SparseMatrix<f64> = upper_submatrix(
        &global_boundary_map_qp1.coo,
        rows_qp1_larger,
        cols_qp1_larger,
    )
    .into();
    // rows_qp1_smaller is codomain of map (q+1) -> (q) of the smaller of the complexes.
    // rows_qp1_larger is codomain of map (q+1) -> (q) of the larger of the complexes.
    let lower_dim_by = rows_qp1_larger - rows_qp1_smaller;
    let up_persistent_laplacian = up_persistent_laplacian(&boundary_map_qp1_larger, lower_dim_by)?;

    Some((up_persistent_laplacian.csc + down_persistent_laplacian.csc).into())
}

// Via computation of _all_ eigenvalues and eigenvectors via Lanczos algorithm for sparse matrices.
pub fn homology_dimension(
    global_boundary_map_qp1: &SparseMatrix<f64>,
    dims_qp1_smaller: (usize, usize),
    dims_qp1_larger: (usize, usize),
    global_boundary_map_q: &SparseMatrix<f64>,
    dims_q_smaller: (usize, usize),
) -> Option<usize> {
    let eigen = eigen_persistent_laplacian(
        global_boundary_map_qp1,
        dims_qp1_smaller,
        dims_qp1_larger,
        global_boundary_map_q,
        dims_q_smaller,
    )?;
    Some(count_zeros(eigen.eigenvalues, 1e-5))
}

pub fn eigen_persistent_laplacian(
    global_boundary_map_qp1: &SparseMatrix<f64>,
    dims_qp1_smaller: (usize, usize),
    dims_qp1_larger: (usize, usize),
    global_boundary_map_q: &SparseMatrix<f64>,
    dims_q_smaller: (usize, usize),
) -> Option<HermitianEigen<f64>> {
    let persistent_laplacian = persistent_laplacian(
        global_boundary_map_qp1,
        dims_qp1_smaller,
        dims_qp1_larger,
        global_boundary_map_q,
        dims_q_smaller,
    )?;
    let eigen = persistent_laplacian.csc.eigsh(50, Order::Smallest);
    Some(eigen)
}

// Dense and slow
fn generalized_schur_complement(M: &DMatrix<f64>, D_rows: usize, D_cols: usize) -> DMatrix<f64> {
    // Dimensions of the global matrix M
    let m = M.nrows();
    let n = M.ncols();

    // Extract blocks based on the provided dimensions of D
    let A_block = M.view((0, 0), (m - D_rows, n - D_rows));
    let B_block = M.view((0, n - D_cols), (m - D_rows, D_cols));
    let C_block = M.view((m - D_rows, 0), (D_rows, n - D_cols));
    let D_block = M.view((m - D_rows, m - D_rows), (D_rows, D_cols));

    // Compute the Moore-Penrose pseudoinverse of D
    let D_pinv = D_block.pseudo_inverse(1e-10).unwrap();

    // Compute the generalized Schur complement
    let BD_pinvC = B_block * D_pinv * C_block;
    let schur = A_block - BD_pinvC;
    schur
}

// Assumes number of (q) simplices increases by at most 1 on each step of filtration
pub fn persistent_laplacians_of_filtration(
    sparse_boundary_maps: HashMap<usize, SparseMatrix<f64>>,
    // filtration_index: {q: dimension_q_simplices}
    filt_hash: HashMap<usize, HashMap<usize, usize>>,
) -> HashMap<usize, HashMap<(usize, usize), usize>> {
    // q: {(K, L): eigenvalues of pair K \hookrightarrow L}
    let mut eigenvalues = HashMap::new();

    // Get the filtration indices in descending order
    let mut filtration_indices: Vec<_> = filt_hash.keys().collect();
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
        let mut q_eigenvalues = HashMap::new();
        for l in &filtration_indices {
            let dimension_hashmap_l = filt_hash.get(l).unwrap();
            let num_q_simplices_l = dimension_hashmap_l.get(&q).unwrap();
            if **l == 9 {
                println!("Num q simplices l: {}", num_q_simplices_l)
            }
            // Only have a boundary map if there are higher dimensional simplices
            let boundary_map_l_qp1: Option<SparseMatrix<f64>> =
                if let Some(num_qp1_simplices_l) = dimension_hashmap_l.get(&(q + 1)) {
                    Some(
                        upper_submatrix(
                            &global_boundary_map_qp1.coo,
                            *num_q_simplices_l,
                            *num_qp1_simplices_l,
                        )
                        .into(),
                    )
                } else {
                    None
                };
            if **l == 9 {
                println!("D2: {:?}", boundary_map_l_qp1);
            }
            let up_laplacian = boundary_map_l_qp1.map(|b| up_laplacian(&b));
            // let up_laplacian: Option<SparseMatrix<f64>> = boundary_map_l_qp1.map(|b| {
            //     let t = b.csc.transpose();
            //     SparseMatrix::from(b.csc * t)
            // });
            if **l == 9 {
                println!("up laplacian before k loop: {:?}", up_laplacian);
            }
            // For each filtration value lower than the current filtration, compute the persistent laplacian.
            // This is the step for which we need the dense filtration.
            for k in (0..=**l).rev() {
                // println!("L, K pair is ({}, {})", l, k);
                // Compute the up persistent laplacian for K \hookrightarrow L inductively
                let dimension_hashmap_k = filt_hash.get(&k).unwrap();
                let num_q_simplices_k = dimension_hashmap_k.get(&q).unwrap();
                if k == 7 && **l == 9 {
                    println!("up laplacian ({:?})", up_laplacian);
                }

                // up_laplacian = up_laplacian.map(|u| {
                //     let mut new_up = u;
                //     let lower_by = new_up.csc.ncols() - num_q_simplices_k;
                //     // We can only lower by 1 at a time, so if lower_by > 1, we take it step by step
                //     for _ in 1..=lower_by {
                //         let new_up_persistent_laplacian =
                //             up_persistent_laplacian_step(new_up).unwrap();
                //         if k == 7 && **l == 9 {
                //             println!(
                //                 "New up has dimensions ({}, {})",
                //                 new_up_persistent_laplacian.coo.nrows(),
                //                 new_up_persistent_laplacian.coo.ncols()
                //             );
                //         }
                //         // Update recursive variable
                //         new_up = new_up_persistent_laplacian;
                //     }
                //     new_up
                // });

                let up_persistent_laplacian = if let Some(up) = up_laplacian.as_ref() {
                    let d_rows = num_q_simplices_l - num_q_simplices_k;
                    let dense_up_laplacian = to_dense(&up.csr);
                    let schur = if d_rows > 0 {
                        generalized_schur_complement(&dense_up_laplacian, d_rows, d_rows)
                    } else {
                        dense_up_laplacian
                    };
                    let schur_coo = CooMatrix::from(&schur);
                    &CsrMatrix::from(&schur_coo)
                    // Compute schur complement
                    // &up.csr
                } else {
                    &CsrMatrix::zeros(*num_q_simplices_k, *num_q_simplices_k)
                };

                // Compute the down persistent laplacian for K \hookrightarrow L
                let num_qm1_simplices_k = dimension_hashmap_k.get(&(q - 1)).unwrap_or(&0);
                let boundary_map_q_k: CsrMatrix<f64> = CsrMatrix::from(&upper_submatrix(
                    &global_boundary_map_q.coo,
                    *num_qm1_simplices_k,
                    *num_q_simplices_k,
                ));
                let down_persistent_laplacian =
                    down_laplacian(&SparseMatrix::from(boundary_map_q_k)).csr;
                // let down_persistent_laplacian = &boundary_map_q_k.transpose() * &boundary_map_q_k;
                let persistent_laplacian = up_persistent_laplacian + down_persistent_laplacian;

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
                if persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0 {
                    let dense = to_dense(&persistent_laplacian);
                    let qr = dense.clone().qr();
                    let rank_defiency =
                        qr.r().diagonal().iter().filter(|d| d.abs() < 1e-12).count();
                    let q = qr.q();
                    let nullspace = q.column(q.ncols() - 1);
                    if k == 7 && **l == 10 {
                        println!("NULLSPACE IS {}", nullspace);
                        println!("NULLSPACE DIM IS {}", rank_defiency);
                        println!("{}", dense * nullspace);
                    }
                    // Compute eigenvalues if the persistent laplacian has dimension > 1
                    // let eigen = if persistent_laplacian.nrows() > 0
                    //     && persistent_laplacian.ncols() > 0
                    // {
                    //     // TODO: why does Lanczos sometimes panic?
                    //     if let Ok(lanczos_result) = catch_unwind(|| {
                    //         let eig = persistent_laplacian.eigsh(500, Order::Smallest).eigenvalues;
                    //         if k == 7 && **l == 10 {
                    //             println!("{:}", eig);
                    //         }
                    //         eig
                    //     }) {
                    //         count_zeros(lanczos_result, 1e-5)
                    //     } else {
                    //         0
                    //     }
                    // } else {
                    //     0
                    // };
                    // // Update results hash
                    q_eigenvalues.insert((k, **l), rank_defiency);
                }
            }
        }
        eigenvalues.insert(*q, q_eigenvalues);
    }
    eigenvalues
}

#[pyfunction]
fn process_tda(py: Python, boundary_maps: &PyDict, filt: &PyDict) -> PyResult<PyObject> {
    let sparse_boundary_maps = process_sparse_dict(boundary_maps).unwrap();

    let filt_hash = parse_nested_dict(&filt).unwrap();
    let eigenvalues = persistent_laplacians_of_filtration(sparse_boundary_maps, filt_hash);
    Ok(eigenvalues.into_py(py))
}

#[pymodule]
fn persistent_laplacians(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_tda, m)?)?;
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

fn drop_last_row_col_coo(matrix: &CooMatrix<f64>) -> CooMatrix<f64> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    upper_submatrix(matrix, nrows - 1, ncols - 1)
}

/// Take upper left submatrix
/// rows and cols are the dimensions of the submatrix to take
/// TODO: maybe take ownership, then early return when rows = nrows, cols = ncols
fn upper_submatrix(matrix: &CooMatrix<f64>, rows: usize, cols: usize) -> CooMatrix<f64> {
    let mut new_coo = CooMatrix::new(rows, cols);
    for (i, j, v) in matrix.triplet_iter() {
        if i < rows && j < cols {
            new_coo.push(i, j, *v);
        }
    }
    new_coo
}

pub fn count_zeros(data: DVector<f64>, eps: f64) -> usize {
    data.iter().filter(|&&x| x.abs() <= eps).count()
}
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
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
        let boundary_map = SparseMatrix::from(coo_boundary);
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
        let up_laplacian = up_laplacian(&boundary_map);
        let persistent_up = up_persistent_laplacian_step(up_laplacian.into()).unwrap();
        let mut expected = CooMatrix::new(2, 2);
        expected.push(0, 0, 1.5);
        expected.push(0, 1, -1.5);
        expected.push(1, 0, -1.5);
        expected.push(1, 1, 1.5);
        assert_eq!(expected, persistent_up.coo)
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
        let boundary = coo_boundary.into();
        let up_laplacian = up_laplacian(&boundary).into();
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

        assert_eq!(
            0.5 * CsrMatrix::from(&expected),
            up_persistent_laplacian.csr
        );
    }

    #[test]
    fn test_homology_example_3_4() {
        let mut coo_boundary_2 = CooMatrix::new(5, 2);
        coo_boundary_2.push(0, 0, 1.0);
        coo_boundary_2.push(1, 0, 1.0);
        coo_boundary_2.push(4, 0, -1.0);

        coo_boundary_2.push(2, 1, 1.0);
        coo_boundary_2.push(3, 1, -1.0);
        coo_boundary_2.push(4, 1, 1.0);
        let boundary_2 = coo_boundary_2.into();

        let mut coo_boundary_1 = CooMatrix::new(4, 5);
        coo_boundary_1.push(0, 0, -1.0);
        coo_boundary_1.push(0, 1, -1.0);
        coo_boundary_1.push(0, 4, -1.0);
        coo_boundary_1.push(1, 0, 1.0);
        coo_boundary_1.push(1, 2, -1.0);
        coo_boundary_1.push(2, 2, 1.0);
        coo_boundary_1.push(2, 3, -1.0);
        coo_boundary_1.push(2, 4, 1.0);
        coo_boundary_1.push(3, 1, 1.0);
        coo_boundary_1.push(3, 3, 1.0);
        let boundary_1 = coo_boundary_1.into();

        let homology_dimension =
            homology_dimension(&boundary_2, (4, 4), (5, 2), &boundary_1, (4, 4));
        assert_eq!(homology_dimension.unwrap(), 1)
    }
}
