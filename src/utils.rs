use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;

/// Checks if a floating-point number is effectively zero within a given tolerance.
/// * `float`: The floating-point number to check.
pub fn is_float_zero(float: f64, tol: f64) -> bool {
    float.abs() < tol
}

/// Extracts the upper submatrix of a CSR matrix, keeping only entries in the first `rows`
/// and first `cols` columns, and ignoring entries below a specified zero tolerance.
/// * `matrix`: The input CSR matrix.
/// * `rows`: The number of rows to keep in the submatrix.
/// * `cols`: The number of columns to keep in the submatrix.
/// * `zero_tol`: The tolerance below which entries are considered zero and ignored.
pub fn upper_submatrix_csr(
    matrix: &CsrMatrix<f64>,
    rows: usize,
    cols: usize,
    zero_tol: f64,
) -> CsrMatrix<f64> {
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
            if j < cols && !is_float_zero(v, zero_tol) {
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
            if j < cols && !is_float_zero(v, zero_tol) {
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

/// Computes the product C[:n]R[:n] of the last column C by the last row R of an n x n matrix,
/// without the last entry, so returning an (n-1) x (n-1) matrix
/// * `csr`: The input CSR matrix, which must be square.
pub fn outer_product_last_col_row_csr(csr: &CsrMatrix<f64>) -> CsrMatrix<f64> {
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

/// Drops the last row and last column of a CSR matrix, returning a new CSR matrix.
/// * `matrix`: The input CSR matrix, which must be square.
/// * `zero_tol`: The tolerance below which entries are considered zero and ignored.
pub fn drop_last_row_col_csr(matrix: &CsrMatrix<f64>, zero_tol: f64) -> CsrMatrix<f64> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    upper_submatrix_csr(matrix, nrows - 1, ncols - 1, zero_tol)
}
