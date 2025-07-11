use nalgebra::zero;
use nalgebra_sparse::na::DMatrix;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use sprs::DenseVector;
use std::cmp::Ordering;
use std::time::Instant;

use crate::sparse::split_csr;
use crate::{
    sparse::{to_dense, SparseMatrix},
    utils::{
        drop_last_row_col_csr, is_float_zero, outer_product_last_col_row_csr, upper_submatrix_csr,
    },
};

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
pub fn up_persistent_laplacian_step(
    prev_up_persistent_laplacian: CsrMatrix<f64>,
    zero_tol: f64,
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
            (is_float_zero(value, zero_tol), value)
        })
        .unwrap_or((true, 0.0));
    let exclude_last_row_col = drop_last_row_col_csr(&prev_up_persistent_laplacian, zero_tol);
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
pub fn compute_up_persistent_laplacian_stepwise(
    num_q_simplices_k: usize,
    up_laplacian: CsrMatrix<f64>,
    zero_tol: f64,
) -> Option<CsrMatrix<f64>> {
    assert!(num_q_simplices_k > 0);
    let lower_by = up_laplacian.ncols() - num_q_simplices_k;
    let mut new_up_persistent_laplacian = up_laplacian;
    // We can only lower by 1 at a time, so if lower_by > 1, we take it step by step
    for _ in 1..=lower_by {
        new_up_persistent_laplacian =
            up_persistent_laplacian_step(new_up_persistent_laplacian, zero_tol).unwrap();
    }
    Some(new_up_persistent_laplacian)
}

fn pseudoinverse(a: DMatrix<f64>, zero_tol: f64) -> Option<DMatrix<f64>> {
    let (m, n) = (a.nrows(), a.ncols());
    // Compute SVD: A = U * Σ * Vᵀ
    let svd = nalgebra_lapack::SVD::new(a)?;
    let u = svd.u;
    let v_t = svd.vt;
    let sigma = svd.singular_values;

    // Tolerance for treating singular values as zero
    let max_sigma = sigma.max();

    // Compute reciprocal of non-zero singular values
    let sigma_inv = sigma.map(|x| if x > zero_tol { 1.0 / x } else { 0.0 });

    // Build Σ⁺ (n × m)
    let sigma_inv_mat = DMatrix::from_diagonal(&sigma_inv);

    // Compute A⁺ = V * Σ⁺ * Uᵀ
    Some(v_t.transpose() * sigma_inv_mat * u.transpose())
}

fn schur_complement_dense(n: usize, m: DMatrix<f64>, zero_tol: f64) -> Option<DMatrix<f64>> {
    let total = m.nrows();
    assert_eq!(total, m.ncols(), "Input must be square");
    assert!(n < total, "n must be less than matrix dimension");
    let p = total - n;
    // Partition blocks
    let a = m.view((0, 0), (n, n)).into_owned();
    let b = m.view((0, n), (n, p)).into_owned();
    let c = m.view((n, 0), (p, n)).into_owned();
    let d = m.view((n, n), (p, p)).into_owned();

    // Compute pseudoinverse of D
    let now = Instant::now();
    let d_pinv = &pseudoinverse(d, zero_tol)?;
    let elapsed = now.elapsed().as_secs_f64();
    println!("SCHUR: pseudoinverse: {elapsed}s");

    // Compute C * A⁺ * B
    let now = Instant::now();
    let bdc = b * d_pinv * c;
    let elapsed = now.elapsed().as_secs_f64();
    println!("SCHUR: mult: {elapsed}s");

    // Schur complement S = D - C A⁺ B
    let now = Instant::now();
    let schur = a - bdc;
    let elapsed = now.elapsed().as_secs_f64();
    println!("SCHUR: sub: {elapsed}s");
    Some(schur)
}

fn schur_complement(n: usize, m: &CsrMatrix<f64>, zero_tol: f64) -> Option<CsrMatrix<f64>> {
    let total = m.nrows();
    assert_eq!(total, m.ncols(), "Input must be square");
    assert!(n < total, "n must be less than matrix dimension");

    // Partition blocks
    let (a, b, c, d) = split_csr(&m, n);
    // Compute pseudoinverse of D
    let d_pinv = CsrMatrix::from(&CooMatrix::from(&pseudoinverse(to_dense(&d), zero_tol)?));
    // Compute C * A⁺ * B
    let bdc = b * d_pinv * c;
    // Schur complement S = D - C A⁺ B
    let schur = a - bdc;
    Some(schur)
}

pub fn compute_up_persistent_laplacian_schur(
    num_q_simplices_k: usize,
    up_laplacian: CsrMatrix<f64>,
    zero_tol: f64,
) -> Option<CsrMatrix<f64>> {
    assert!(num_q_simplices_k > 0);
    if num_q_simplices_k == up_laplacian.ncols() {
        return Some(up_laplacian.clone());
    }
    // TODO: now fails quietly when SVD computation fails
    let schur = schur_complement(num_q_simplices_k, &up_laplacian, zero_tol);
    if schur.is_none() {
        println!("Failed SVD computation");
    }
    Some(schur?)
}

fn compute_down_persistent_laplacian(
    num_qm1_simplices_k: usize,
    num_q_simplices_k: usize,
    global_boundary_map_q: &SparseMatrix<f64>,
    zero_tol: f64,
) -> SparseMatrix<f64> {
    let boundary_map_q_k: CsrMatrix<f64> = upper_submatrix_csr(
        &global_boundary_map_q.csr,
        num_qm1_simplices_k,
        num_q_simplices_k,
        zero_tol,
    );
    let down_persistent_laplacian = down_laplacian(&SparseMatrix::from(boundary_map_q_k));
    down_persistent_laplacian
}

pub fn compute_down_persistent_laplacian_transposing(
    num_qm1_simplices_k: usize,
    num_q_simplices_k: usize,
    global_boundary_map_q: &CsrMatrix<f64>,
    zero_tol: f64,
) -> CsrMatrix<f64> {
    let q_boundary_k = upper_submatrix_csr(
        &global_boundary_map_q,
        num_qm1_simplices_k,
        num_q_simplices_k,
        zero_tol,
    );
    let down_persistent_laplacian = down_laplacian_transposing(&q_boundary_k);
    down_persistent_laplacian
}

#[cfg(test)]
mod tests {
    use crate::sparse::to_dense;

    use super::*;
    use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
    use rand::{rngs::StdRng, Rng, SeedableRng};
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
        let persistent_up = up_persistent_laplacian_step(up_laplacian.into(), 1e-6).unwrap();
        let mut expected = CooMatrix::new(2, 2);
        expected.push(0, 0, 1.5);
        expected.push(0, 1, -1.5);
        expected.push(1, 0, -1.5);
        expected.push(1, 1, 1.5);
        assert_eq!(expected, CooMatrix::from(&persistent_up))
    }

    #[test]
    fn test_up_persistent_laplacian_2_path() {
        let mut coo_boundary = CooMatrix::new(4, 5);
        coo_boundary.push(0, 0, -1.0);
        coo_boundary.push(0, 1, -1.0);
        coo_boundary.push(1, 2, -1.0);
        coo_boundary.push(1, 3, -1.0);
        coo_boundary.push(2, 1, 1.0);
        coo_boundary.push(2, 2, 1.0);
        coo_boundary.push(2, 4, -1.0);
        coo_boundary.push(3, 3, 1.0);
        coo_boundary.push(3, 4, 1.0);
        let boundary_map = CsrMatrix::from(&coo_boundary);
        let up_laplacian = up_laplacian_transposing(&boundary_map);
        let pers_stepwise =
            compute_up_persistent_laplacian_stepwise(3, up_laplacian.clone(), 1e-6).unwrap();
        let pers_schur =
            compute_up_persistent_laplacian_schur(3, up_laplacian.clone(), 1e-6).unwrap();
        assert_eq!(pers_stepwise, pers_schur);
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
        let up_persistent_laplacian = up_persistent_laplacian_step(up_laplacian, 1e-6).unwrap();

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

    #[test]
    fn test_pseudoinverse() {
        // Example matrix (2 × 3)
        let a =
            DMatrix::<f64>::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let pinv = pseudoinverse(a.clone(), 1e-6).unwrap();
        // Check Penrose condition A * A⁺ * A ≈ A
        let approx = &a * &pinv * &a;
        assert!(approx.relative_eq(&a, 1e-6, 1e-6));
    }

    /// Make a random symmetric "Laplacian‑like" CsrMatrix of size n×n.
    fn random_laplacian(n: usize, seed: u64) -> CsrMatrix<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            // initial diagonal
            let diag: f64 = rng.random_range(1.0..5.0);
            coo.push(i, i, diag);
            if i + 1 < n {
                let w: f64 = rng.random_range(0.0..1.0);
                // off‑diagonals
                coo.push(i, i + 1, -w);
                coo.push(i + 1, i, -w);
                // bump diagonals by w
                coo.push(i, i, w);
                coo.push(i + 1, i + 1, w);
            }
        }
        CsrMatrix::from(&coo)
    }

    /// Assert two sparse mats approx‑equal via their dense forms.
    fn assert_dense_approx_eq(a: &CsrMatrix<f64>, b: &CsrMatrix<f64>, tol: f64) {
        let da = to_dense(a);
        let db = to_dense(b);
        println!("DA shape: {:?}, DB shape: {:?}", da.shape(), db.shape());
        assert_eq!(da.shape(), db.shape(), "shape mismatch");
        assert!(da.relative_eq(&db, tol, tol));
    }

    #[test]
    fn no_op_when_k_eq_n() {
        let lap = random_laplacian(5, 42);
        let step = compute_up_persistent_laplacian_stepwise(5, lap.clone(), 1e-6).unwrap();
        let schur = compute_up_persistent_laplacian_schur(5, lap.clone(), 1e-6).unwrap();
        assert_dense_approx_eq(&step, &lap, 1e-12);
        assert_dense_approx_eq(&schur, &lap, 1e-12);
    }

    #[test]
    fn stepwise_vs_schur_random() {
        for &(n, k, seed) in &[
            (6, 3, 0),
            (7, 5, 1),
            (8, 2, 123),
            (10, 7, 999),
            (1500, 400, 42),
            (1500, 1000, 43),
        ] {
            let base = random_laplacian(n, seed);
            let step = compute_up_persistent_laplacian_stepwise(k, base.clone(), 1e-6)
                .expect("stepwise failed");
            let schur =
                compute_up_persistent_laplacian_schur(k, base.clone(), 1e-6).expect("schur failed");
            assert_dense_approx_eq(&step, &schur, 1e-8);
        }
    }

    #[test]
    #[should_panic]
    fn invalid_k_zero_panics() {
        let lap = random_laplacian(4, 7);
        let _ = compute_up_persistent_laplacian_stepwise(0, lap, 1e-6);
    }
}
