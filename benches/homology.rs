use criterion::{criterion_group, criterion_main, Criterion};
use persistent_laplacians::{
    parse_nested_dict, persistent_laplacians_of_filtration, process_sparse_dict,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};

fn bench_process_tda(c: &mut Criterion) {
    let mut group = c.benchmark_group("homology");
    group.sample_size(10);
    // For now, we use the python code that takes datapoints and makes us a filtration
    let (sparse_boundary_maps, filt_hash) = Python::with_gil(|py| {
        let sys = py.import("sys").unwrap();
        let path: &PyList = sys.getattr("path").unwrap().downcast().unwrap();
        path.insert(0, ".venv/lib/python3.10/site-packages")
            .unwrap();
        path.insert(0, "python").unwrap();

        let data_module = PyModule::import(py, "persistent_laplacians.data").unwrap();
        let sphere_data = data_module
            .getattr("sphere_data")
            .unwrap()
            .call1((30, 2, 1.0_f64, 0.0_f64, 42_u64))
            .unwrap();

        let tup: &PyTuple = sphere_data.downcast().unwrap();
        let mats_py = tup.get_item(0).unwrap();
        let fmap_py = tup.get_item(1).unwrap();

        let mats: &PyDict = mats_py.extract().unwrap();
        let fmap: &PyDict = fmap_py.extract().unwrap();

        let sparse_boundary_maps = process_sparse_dict(mats).unwrap();
        let filt_hash = parse_nested_dict(fmap).unwrap();
        (sparse_boundary_maps, filt_hash)
    });
    group.bench_function("process_tda(30,2,1.0,0.0,42)", |b| {
        b.iter_batched(
            || (sparse_boundary_maps.clone(), filt_hash.clone()),
            |(maps, hash)| {
                let eigenvalues = persistent_laplacians_of_filtration(maps, hash);
                criterion::black_box(eigenvalues);
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_process_tda);
criterion_main!(benches);
