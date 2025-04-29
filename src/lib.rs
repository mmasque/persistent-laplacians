use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sprs::DenseVector;
use std::cmp::Ordering;
use std::collections::HashMap;

// Temporary before I find a better solution.
fn is_float_zero(float: f64) -> bool {
    float.abs() < 1e-15
}

// Sometimes you want them all
#[derive(Debug, Clone)]
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

fn up_laplacian_transposing(boundary_map_qp1: &SparseMatrix<f64>) -> SparseMatrix<f64> {
    SparseMatrix::from(&boundary_map_qp1.csr * boundary_map_qp1.csr.transpose())
}

fn down_laplacian(boundary_map_q: &SparseMatrix<f64>) -> SparseMatrix<f64> {
    let degree_matrix = down_degree(boundary_map_q);
    let adjacency_matrix = down_adjacency(boundary_map_q);
    (degree_matrix.csr - adjacency_matrix.csr).into()
}

fn down_laplacian_transposing(boundary_map_q: &SparseMatrix<f64>) -> SparseMatrix<f64> {
    SparseMatrix::from(boundary_map_q.csr.transpose() * &boundary_map_q.csr)
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
    let exclude_last_row_col_coo = drop_last_row_col_coo(&coo);
    if new_dim > 82 && new_dim < 85 {
        println!("bottom right with new dimension {new_dim} is: {bottom_right_value}");
    }
    if !bottom_right_is_zero {
        // prev(i, j) - prev(i, dim_prev) * prev(dim_prev, j) / prev(dim_prev, dim_prev)
        let outer_coo = outer_product_last_col_row(&prev_up_persistent_laplacian);
        let outer_weighed = CsrMatrix::from(&outer_coo) / bottom_right_value;
        let exclude_last_row_col = CsrMatrix::from(&exclude_last_row_col_coo);
        return Some(SparseMatrix::from(exclude_last_row_col - outer_weighed));
    } else {
        return Some(SparseMatrix::from(exclude_last_row_col_coo));
    }
}

// Dense and slow
fn generalized_schur_complement(M: &DMatrix<f64>, D_rows: usize, D_cols: usize) -> DMatrix<f64> {
    // Dimensions of the global matrix M
    let m = M.nrows();
    let n = M.ncols();

    let A_block = M.view((0, 0), (m - D_rows, n - D_cols));
    let B_block = M.view((0, n - D_cols), (m - D_rows, D_cols));
    let C_block = M.view((m - D_rows, 0), (D_rows, n - D_cols));
    let D_block = M.view((m - D_rows, n - D_cols), (D_rows, D_cols));

    // Compute the Moore-Penrose pseudoinverse of D
    let D_pinv = D_block.pseudo_inverse(1e-10).unwrap();

    // Compute the generalized Schur complement
    let BD_pinvC = B_block * D_pinv * C_block;
    let schur = A_block - BD_pinvC;
    schur
}

fn compute_up_persistent_laplacian_schur(
    up_laplacian: &SparseMatrix<f64>,
    num_q_simplices_k: usize,
) -> DMatrix<f64> {
    assert!(num_q_simplices_k > 0);
    let bottom_diag_shape = up_laplacian.csc.ncols() - num_q_simplices_k;
    if bottom_diag_shape > 0 {
        generalized_schur_complement(
            &to_dense(&up_laplacian.csr),
            bottom_diag_shape,
            bottom_diag_shape,
        )
    } else {
        to_dense(&up_laplacian.csr)
    }
}

/// Computes the qth up persistent laplacian of a pair of simplicial complexes K hookrightarrow L given the qth up laplacian
/// of L and the number of q simplices of K.
fn compute_up_persistent_laplacian(
    num_q_simplices_k: usize,
    up_laplacian: SparseMatrix<f64>,
) -> SparseMatrix<f64> {
    assert!(num_q_simplices_k > 0);
    let lower_by = up_laplacian.csc.ncols() - num_q_simplices_k;
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

fn compute_down_persistent_laplacian_transposing(
    num_qm1_simplices_k: usize,
    num_q_simplices_k: usize,
    global_boundary_map_q: &SparseMatrix<f64>,
) -> SparseMatrix<f64> {
    let q_boundary_k = CsrMatrix::from(&upper_submatrix(
        &global_boundary_map_q.coo,
        num_qm1_simplices_k,
        num_q_simplices_k,
    ));
    let down_persistent_laplacian = down_laplacian_transposing(&SparseMatrix::from(q_boundary_k));
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

fn compute_homology_from_persistent_laplacian(persistent_laplacian: &CsrMatrix<f64>) -> usize {
    assert!(persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0);
    let dense = to_dense(&persistent_laplacian);
    compute_homology_from_persistent_laplacian_dense(&dense)
}

fn compute_homology_from_persistent_laplacian_dense(persistent_laplacian: &DMatrix<f64>) -> usize {
    assert!(persistent_laplacian.nrows() > 0 && persistent_laplacian.ncols() > 0);
    let qr = persistent_laplacian.clone().qr();
    let rank_deficiency = qr.r().diagonal().iter().filter(|d| d.abs() < 1e-12).count();
    rank_deficiency
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
            if num_q_simplices_l == &0 {
                // The map is always zero, so we don't care
                continue;
            }
            // Only have a boundary map if there are higher dimensional simplices
            let num_qp1_simplices_l = dimension_hashmap_l.get(&(q + 1)).unwrap_or(&0);
            let boundary_map_l_qp1: Option<SparseMatrix<f64>> = if num_qp1_simplices_l > &0 {
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
            let mut up_persistent_laplacian =
                boundary_map_l_qp1.map(|b| up_laplacian_transposing(&b));
            // For each filtration value lower than the current filtration, compute the persistent laplacian.
            // let up_laplacian = boundary_map_l_qp1.map(|b| up_laplacian_transposing(&b));
            for k in (0..=**l).rev() {
                let dimension_hashmap_k = filt_hash.get(&k).unwrap();
                let num_q_simplices_k = dimension_hashmap_k.get(&q).unwrap();
                if num_q_simplices_k == &0 {
                    // The map is always zero, so we don't care
                    continue;
                }
                let num_qm1_simplices_k = dimension_hashmap_k.get(&(q - 1)).unwrap_or(&0);
                if l == &&83 && *q == 1 {
                    println!("K = {k}, L = {l}");
                    println!("Num 0-simplices of K: {}", num_qm1_simplices_k);
                    println!("Num 1-simplices of K: {}", num_q_simplices_k);
                    // println!(
                    //     "Dimensions of previous up persistent laplacian: ({}, {})",
                    //     up_persistent_laplacian.as_ref().unwrap().csc.ncols(),
                    //     up_persistent_laplacian.as_ref().unwrap().csc.ncols()
                    // );
                    println!("Num 2-simplices of L: {}", num_qp1_simplices_l);
                    println!("Num 1-simplices of L: {}", num_q_simplices_l);
                }
                // Compute the up persistent laplacian for K \hookrightarrow L inductively
                up_persistent_laplacian = up_persistent_laplacian
                    .map(|u| compute_up_persistent_laplacian(*num_q_simplices_k, u));

                // Compute the down persistent laplacian for K \hookrightarrow L
                // If there are no lower simplices, the map factors via the 0 vector space, so it is zero
                let down_persistent_laplacian = if num_qm1_simplices_k > &0 {
                    Some(compute_down_persistent_laplacian_transposing(
                        *num_qm1_simplices_k,
                        *num_q_simplices_k,
                        &global_boundary_map_q,
                    ))
                } else {
                    None
                };

                if let Some(persistent_laplacian) =
                    match (&up_persistent_laplacian, &down_persistent_laplacian) {
                        (Some(up), Some(down)) => Some(&up.csr + &down.csr),
                        (None, None) => None,
                        (Some(up), None) => Some(up.csr.clone()),
                        (None, Some(down)) => Some(down.csr.clone()),
                    }
                {
                    let homology =
                        compute_homology_from_persistent_laplacian(&persistent_laplacian);
                    q_eigenvalues.insert((k, **l), homology);
                }

                // if let Some(persistent_laplacian) =
                //     match (&up_persistent_laplacian, &down_persistent_laplacian) {
                //         (Some(up), Some(down)) => Some(up + to_dense(&down.csr)),
                //         (None, None) => None,
                //         (Some(up), None) => Some(up.clone()),
                //         (None, Some(down)) => Some(to_dense(&down.csr)),
                //     }
                // {
                //     let homology =
                //         compute_homology_from_persistent_laplacian_dense(&persistent_laplacian);
                //     q_eigenvalues.insert((k, **l), homology);
                // }
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

    use approx::assert_relative_eq;
    use nalgebra::DMatrix;

    /// Your Schur‐complement function under test
    use super::generalized_schur_complement;

    #[test]
    fn test_generalized_schur_complement_2x2_blocks() {
        // Construct A,B,C,D as small 2×2 matrices
        let A = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let B = DMatrix::from_row_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        let C = DMatrix::from_row_slice(2, 2, &[9.0, 10.0, 11.0, 12.0]);
        let D = DMatrix::from_row_slice(2, 2, &[13.0, 14.0, 15.0, 16.0]);

        // Build the full 4×4 M = [A B; C D]
        let mut data = Vec::with_capacity(16);
        data.extend(A.iter());
        data.extend(B.iter());
        data.extend(C.iter());
        data.extend(D.iter());
        let M = DMatrix::from_row_slice(4, 4, &data);

        // Compute explicit Schur: A - B * D⁻¹ * C
        let D_inv = D.clone().pseudo_inverse(1e-10).unwrap();
        let expected = &A - &B * &D_inv * &C;

        // Call your helper (eliminate last p=2 rows/cols)
        let result = generalized_schur_complement(&M, 2, 2);

        // Compare with a small tolerance
        assert_relative_eq!(result, expected, epsilon = 1e-8);
    }
}
