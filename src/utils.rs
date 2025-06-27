use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CsrMatrix};

use crate::TOL;

// Temporary before I find a better solution.
pub fn is_float_zero(float: f64) -> bool {
    float.abs() < TOL
}

/// Take upper left submatrix
/// rows and cols are the dimensions of the submatrix to take
/// TODO: maybe take ownership, then early return when rows = nrows, cols = ncols
pub fn upper_submatrix(matrix: &CooMatrix<f64>, rows: usize, cols: usize) -> CooMatrix<f64> {
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
