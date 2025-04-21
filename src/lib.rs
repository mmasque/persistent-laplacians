use std::collections::HashMap;

use nalgebra_sparse::csr::CsrMatrix;
use nalgebra_sparse::CooMatrix;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sprs::prod;

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

fn process_sparse_dict(dict: &PyDict) -> PyResult<HashMap<usize, CsrMatrix<f64>>> {
    let mut result = HashMap::new();

    for (key, value) in dict.iter() {
        // Extract CSR components with zero-copy numpy array access
        let indptr = unsafe {
            value
                .getattr("indptr")?
                .downcast::<PyArray1<usize>>()?
                .as_slice()?
                .to_vec()
        };

        let indices = unsafe {
            value
                .getattr("indices")?
                .downcast::<PyArray1<usize>>()?
                .as_slice()?
                .to_vec()
        };

        let data = unsafe {
            value
                .getattr("data")?
                .downcast::<PyArray1<f64>>()?
                .as_slice()?
                .to_vec()
        };

        // Calculate matrix dimensions
        let nrows = indptr
            .len()
            .checked_sub(1)
            .ok_or(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Empty indptr array",
            ))?;

        let ncols = indices.iter().max().map(|&m| m + 1).unwrap_or(0);

        // Build CSR matrix
        let csr = CsrMatrix::try_from_csr_data(nrows, ncols, indptr, indices, data)
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("CSR error"))?;

        result.insert(key.extract::<usize>()?, csr);
    }

    Ok(result)
}

fn up_degree(boundary_map_qp1: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    // start with a zero matrix. Dimensions of codomain in boundary map out of q+1 simplices
    let domain_dimension = boundary_map_qp1.nrows();

    // Degree computation
    // for each q-simplex i (think vertex), look at each q+1 simplex j (think edge) containing it
    // j contains i iff boundary_map_qp1(i, j) != 0
    // So the number of nonzero entries in the ith row is what we want. This is easy using row_offsets
    let row_offsets = boundary_map_qp1.row_offsets();
    let mut degrees = Vec::with_capacity(domain_dimension);
    for i in 0..domain_dimension {
        degrees.push((row_offsets[i + 1] - row_offsets[i]) as f64);
    }
    let mut indptr = Vec::with_capacity(domain_dimension + 1);
    let mut indices = Vec::with_capacity(domain_dimension);
    indptr.push(0);
    for (i, _) in degrees.iter().enumerate() {
        indptr.push(i + 1);
        indices.push(i);
    }
    let degree_matrix =
        CsrMatrix::try_from_csr_data(domain_dimension, domain_dimension, indptr, indices, degrees)
            .expect("Failed to construct degree matrix: invalid CSR parameters");

    return degree_matrix;
}

fn up_adjacency(boundary_map_qp1: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    // For each (q+1) simplex l look at all its q simplices
    // For each pair (i, j) of q simplices of the q+1 simplex
    // boundary_map_qp1(i, l) says if l contains i, with +-1.
    // boundary_map_qp1(j, l) says if l contains j, with +-1.
    let n = boundary_map_qp1.nrows();
    let mut adjacency = HashMap::new();
    let mut column_map = HashMap::new();
    for row in 0..boundary_map_qp1.nrows() {
        let row_slice = boundary_map_qp1.get_row(row).unwrap();
        for (&col, &val) in row_slice.col_indices().iter().zip(row_slice.values()) {
            column_map
                .entry(col)
                .or_insert_with(Vec::new)
                .push((row, val));
        }
    }

    // column_map has, for each q+1 simplex j, a vector (i, B(i, j)) for the i indices for which B(i, j) is nonzero
    // for each q+1 simplex we look at this vector.
    // for each entry pair (row_i, val_i), (row_j, val_j) we
    for (_, entries) in column_map {
        let entry_count = entries.len();
        for i in 0..entry_count {
            let (row_i, val_i) = entries[i];
            for j in (i + 1)..entry_count {
                let (row_j, val_j) = entries[j];
                let product = -1.0 * val_i * val_j;

                // Update adjacency matrix entries
                adjacency.insert((row_i, row_j), product);
                adjacency.insert((row_j, row_i), product);
            }
        }
    }

    let mut coo = CooMatrix::new(n, n);
    for ((i, j), val) in adjacency {
        coo.push(i, j, val);
    }

    CsrMatrix::from(&coo)
}

fn up_laplacian(boundary_map_qp1: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let degree_matrix = up_degree(boundary_map_qp1);
    let adjacency_matrix = up_adjacency(boundary_map_qp1);
    degree_matrix - adjacency_matrix
}

fn down_laplacian(
    boundary_map: CsrMatrix<f64>,
    domain_index: usize,
    codomain_index: usize,
) -> CsrMatrix<f64> {
    todo!()
}

#[pyfunction]
fn process_tda(py: Python, boundary_maps: &PyDict, filt: &PyDict) -> PyResult<PyObject> {
    let sparse_boundary_maps = process_sparse_dict(boundary_maps).unwrap();
    let filt_hash = parse_nested_dict(&filt).unwrap();

    Ok(0.into_py(py))
}

#[pymodule]
fn persistent_laplacians(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_tda, m)?)?;
    Ok(())
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
        // We can add elements in any order. For clarity, we do so in row-major order here.
        coo_boundary.push(0, 0, -1.0);
        coo_boundary.push(0, 2, -1.0);
        coo_boundary.push(1, 0, 1.0);
        coo_boundary.push(1, 1, -1.0);
        coo_boundary.push(2, 1, 1.0);
        coo_boundary.push(2, 2, 1.0);
        let boundary_map_csr = CsrMatrix::from(&coo_boundary);
        let lap = up_laplacian(&boundary_map_csr);

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
        let expected_data = CsrMatrix::from(&expected_data_coo);
        assert_eq!(expected_data, lap)
    }

    #[test]
    fn test_up_laplacian_2simplex() {
        let mut coo_boundary = CooMatrix::new(3, 1);
        // We can add elements in any order. For clarity, we do so in row-major order here.
        coo_boundary.push(0, 0, 1.0);
        coo_boundary.push(1, 0, 1.0);
        coo_boundary.push(2, 0, -1.0);
        let boundary_map_csr = CsrMatrix::from(&coo_boundary);
        let lap = up_laplacian(&boundary_map_csr);

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
        let expected_data = CsrMatrix::from(&expected_data_coo);
        assert_eq!(expected_data, lap)
    }
}
