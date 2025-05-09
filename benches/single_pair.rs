use criterion::{criterion_group, criterion_main, Criterion};
use persistent_laplacians::homology::{
    compute_homology_from_persistent_laplacian_dense,
    compute_homology_from_persistent_laplacian_eigenvalues,
    compute_homology_from_persistent_laplacian_lanczos_crate,
    compute_homology_from_persistent_laplacian_scipy, count_nnz_persistent_laplacian,
    ScipyEigshConfig,
};
mod helpers;
use persistent_laplacians::{persistent_laplacians_of_filtration, PersistentLaplaciansConfig};
use pyo3::{
    types::{PyList, PyModule},
    Python,
};

fn bench_process_single_pair(c: &mut Criterion) {
    let ns = [100, 200, 400, 800];
    let ds = [1];
    for &d in &ds {
        let group_name = format!(
            "Persistent laplacian computation for pair of simplices from {}-spheres",
            d
        );
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(10);
        for &n in &ns {
            let (sparse_boundary_maps, filt_hash) = helpers::boundary_maps_sphere_from_python(n, d);
            let largest_index = *filt_hash.keys().max().unwrap();
            let config = Some(PersistentLaplaciansConfig {
                filtration_indices: vec![largest_index, largest_index - 1],
            });
            group.bench_with_input(
                criterion::BenchmarkId::new("no homology, one pair", n),
                &n,
                |b, &_n| {
                    b.iter_batched(
                        || (sparse_boundary_maps.clone(), filt_hash.clone()),
                        |(maps, hash)| {
                            let eigenvalues = persistent_laplacians_of_filtration(
                                maps,
                                hash,
                                count_nnz_persistent_laplacian,
                                config.clone(),
                            );
                            criterion::black_box(eigenvalues);
                        },
                        criterion::BatchSize::SmallInput,
                    )
                },
            );

            group.bench_with_input(
                criterion::BenchmarkId::new("lanczos, one pair", n),
                &n,
                |b, &_n| {
                    b.iter_batched(
                        || (sparse_boundary_maps.clone(), filt_hash.clone()),
                        |(maps, hash)| {
                            let eigenvalues = persistent_laplacians_of_filtration(
                                maps,
                                hash,
                                compute_homology_from_persistent_laplacian_lanczos_crate,
                                config.clone(),
                            );
                            criterion::black_box(eigenvalues);
                        },
                        criterion::BatchSize::SmallInput,
                    )
                },
            );

            Python::with_gil(|py| {
                let sys = py.import("sys").unwrap();
                let path: &PyList = sys.getattr("path").unwrap().downcast().unwrap();
                path.insert(0, ".venv/lib/python3.10/site-packages")
                    .unwrap();
                let eigsh = PyModule::import(py, "scipy.sparse.linalg")
                    .unwrap()
                    .getattr("eigsh")
                    .unwrap();
                let scipy_sparse = PyModule::import(py, "scipy.sparse").unwrap();

                group.bench_with_input(criterion::BenchmarkId::new("scipy", n), &n, |b, &_n| {
                    b.iter_batched(
                        || {
                            let scipy_config = ScipyEigshConfig::new(
                                py,
                                2,
                                Some(0.00001),
                                None,
                                Some(50),
                                "LM",
                                &eigsh,
                                &scipy_sparse,
                            );
                            return (
                                sparse_boundary_maps.clone(),
                                filt_hash.clone(),
                                scipy_config,
                            );
                        },
                        |(maps, hash, scipy_config)| {
                            let eigenvalues = persistent_laplacians_of_filtration(
                                maps,
                                hash,
                                |p| {
                                    compute_homology_from_persistent_laplacian_scipy(
                                        p,
                                        &scipy_config,
                                    )
                                },
                                config.clone(),
                            );
                            criterion::black_box(eigenvalues);
                        },
                        criterion::BatchSize::SmallInput,
                    )
                });
            });
        }
    }
}

criterion_group!(benches, bench_process_single_pair);
criterion_main!(benches);
