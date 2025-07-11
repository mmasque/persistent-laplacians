use std::collections::HashMap;

use persistent_laplacians::{parse_nested_dict, process_sparse_dict, sparse::SparseMatrix};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};

pub const TOL: f64 = 1e-6;
pub fn boundary_maps_sphere_from_python(
    n: usize,
    d: usize,
) -> (
    HashMap<usize, SparseMatrix<f64>>,
    HashMap<usize, HashMap<usize, usize>>,
) {
    Python::with_gil(|py| {
        let sys = py.import("sys").unwrap();
        let path: &PyList = sys.getattr("path").unwrap().downcast().unwrap();
        path.insert(0, ".venv/lib/python3.10/site-packages")
            .unwrap();
        path.insert(0, "python").unwrap();

        let data_module = PyModule::import(py, "persistent_laplacians.data").unwrap();
        let sphere_data = data_module
            .getattr("sphere_data")
            .unwrap()
            .call1((n, d, 1.0_f64, 0.0_f64, 42_u64))
            .unwrap();

        let tup: &PyTuple = sphere_data.downcast().unwrap();
        let mats_py = tup.get_item(0).unwrap();
        let fmap_py = tup.get_item(1).unwrap();

        let mats: &PyDict = mats_py.extract().unwrap();
        let fmap: &PyDict = fmap_py.extract().unwrap();

        let sparse_boundary_maps = process_sparse_dict(mats).unwrap();
        let filt_hash = parse_nested_dict(fmap).unwrap();
        (sparse_boundary_maps, filt_hash)
    })
}
