use nalgebra::DMatrix;
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
