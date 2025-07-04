use std::{cmp::Ordering, time::Instant};

use csv::Writer;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use sprs::DenseVector;

use crate::{
    sparse::SparseMatrix,
    utils::{
        drop_last_row_col_csr, is_float_zero, outer_product_last_col_row_csr, upper_submatrix,
        upper_submatrix_csr,
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

pub fn up_persistent_laplacian_step_timer(
    prev: CsrMatrix<f64>,
    mut csv_wtr: &mut Writer<std::fs::File>,
) -> Option<CsrMatrix<f64>> {
    let mut row = vec![];

    // 1) get dims
    let t0 = Instant::now();
    let dim_prev = prev.ncols();
    let new_dim = dim_prev.checked_sub(1)?;
    row.push(("get_dims", t0.elapsed().as_secs_f64()));

    // 2) read bottom‑right
    let t1 = Instant::now();
    let (is_zero, br_val) = prev
        .get_entry(new_dim, new_dim)
        .map(|x| {
            let v = x.into_value();
            (is_float_zero(v), v)
        })
        .unwrap_or((true, 0.0));
    row.push(("read_br", t1.elapsed().as_secs_f64()));

    // 3) drop last row/col
    let t2 = Instant::now();
    let base = drop_last_row_col_csr(&prev);
    row.push(("drop_rc", t2.elapsed().as_secs_f64()));

    // 4) maybe compute outer
    let result = if !is_zero {
        let t3 = Instant::now();
        let outer = outer_product_last_col_row_csr(&prev) / br_val;
        row.push(("compute_outer", t3.elapsed().as_secs_f64()));

        let t4 = Instant::now();
        let m = &base - &outer;
        row.push(("subtract", t4.elapsed().as_secs_f64()));
        m
    } else {
        base
    };

    // write one CSV row: step_name,time_s
    for (step, t) in row {
        csv_wtr.write_record(&[step, &format!("{:.9}", t)]).ok();
    }
    csv_wtr.flush().ok();

    Some(result)
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
    println!("LOWER BY: {lower_by}");
    let mut wtr = Writer::from_path(format!(
        "timings_up_{}_{}.csv",
        &new_up_persistent_laplacian.ncols(),
        num_q_simplices_k
    ))
    .unwrap();
    wtr.write_record(&["step", "time_s"]).unwrap();
    for _ in 1..=lower_by {
        new_up_persistent_laplacian =
            up_persistent_laplacian_step_timer(new_up_persistent_laplacian, &mut wtr).unwrap();
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

#[cfg(test)]
mod tests {
    use crate::{
        eigenvalues::compute_eigenvalues_from_persistent_laplacian_primme_crate, sparse::to_dense,
    };

    use super::*;
    use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
    use pyo3::{
        types::{PyList, PyModule},
        Python,
    };
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
        let pers = compute_up_persistent_laplacian(3, up_laplacian);
        let eigs = compute_eigenvalues_from_persistent_laplacian_primme_crate(&pers, 3);
        let dense = to_dense(&pers);
        println!("MATRIX: {}", dense);
        println!("EIGS: {:?}", eigs);
        println!("DENSE EIGS: {:?}", dense.eigenvalues())
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
}
