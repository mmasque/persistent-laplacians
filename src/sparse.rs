use nalgebra_sparse::na::DMatrix;
use nalgebra_sparse::{CooMatrix, CscMatrix, CsrMatrix};

// Sometimes you want them all
#[derive(Debug, Clone)]
pub struct SparseMatrix<T> {
    pub csc: CscMatrix<T>,
    pub csr: CsrMatrix<T>,
    pub coo: CooMatrix<T>,
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

pub fn to_dense(csr: &CsrMatrix<f64>) -> DMatrix<f64> {
    let nrows = csr.nrows();
    let ncols = csr.ncols();
    let mut dense = DMatrix::<f64>::zeros(nrows, ncols);

    // iterate all stored (i, j, v) triplets and write into the dense matrix
    for (i, j, v) in csr.triplet_iter() {
        dense[(i, j)] = *v;
    }
    dense
}

pub fn split_csr(
    m: &CsrMatrix<f64>,
    n: usize,
) -> (
    CsrMatrix<f64>,
    CsrMatrix<f64>,
    CsrMatrix<f64>,
    CsrMatrix<f64>,
) {
    let nrows = m.nrows();
    let ncols = m.ncols();
    assert_eq!(ncols, nrows, "m must be square");
    assert!(n <= nrows, "n must be ≤ matrix size");

    let rows = [0..n, 0..n, n..nrows, n..nrows];
    let cols = [0..n, n..nrows, 0..n, n..nrows];

    let mut blocks = Vec::with_capacity(4);

    for (r_range, c_range) in rows.iter().zip(cols.iter()) {
        let mut triplets = Vec::new();

        for i in r_range.clone() {
            let row = m.row(i);
            for (j, v) in row.col_indices().iter().zip(row.values()) {
                if c_range.contains(&j) {
                    let local_i = i - r_range.start;
                    let local_j = j - c_range.start;
                    triplets.push((local_i, local_j, *v));
                }
            }
        }

        let nrows = r_range.len();
        let ncols = c_range.len();
        let mut coo = CooMatrix::new(nrows, ncols);
        for (i, j, v) in triplets {
            coo.push(i, j, v);
        }
        let csr = CsrMatrix::from(&coo);
        blocks.push(csr);
    }

    let a = blocks.remove(0);
    let b = blocks.remove(0);
    let c = blocks.remove(0);
    let d = blocks.remove(0);
    (a, b, c, d)
}
