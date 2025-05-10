use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, Criterion};
use lanczos::Hermitian;
use nalgebra_sparse::CsrMatrix;
use persistent_laplacians::{
    compute_down_persistent_laplacian_transposing, compute_up_persistent_laplacian,
    homology::{
        compute_homology_from_persistent_laplacian_dense,
        compute_homology_from_persistent_laplacian_eigenvalues,
        compute_homology_from_persistent_laplacian_lanczos_crate,
        compute_homology_from_persistent_laplacian_scipy, count_nnz_persistent_laplacian,
        ScipyEigshConfig,
    },
    up_laplacian_transposing, SparseMatrix,
};
mod helpers;
use pyo3::{
    types::{PyList, PyModule},
    Python,
};

fn single_pair_persistent_laplacian(
    sparse_boundary_maps: HashMap<usize, SparseMatrix<f64>>,
    filt_hash: HashMap<usize, HashMap<usize, usize>>,
    k: usize,
) -> CsrMatrix<f64> {
    let boundary_2 = sparse_boundary_maps.get(&2).unwrap();
    let boundary_1 = sparse_boundary_maps.get(&1).unwrap();
    let num_1_simplices_k = filt_hash.get(&k).unwrap().get(&1).unwrap();
    let num_0_simplices_k = filt_hash.get(&k).unwrap().get(&0).unwrap();

    let up = up_laplacian_transposing(&boundary_2.csr);
    let up_persistent = compute_up_persistent_laplacian(*num_1_simplices_k, up);

    let down_persistent = compute_down_persistent_laplacian_transposing(
        *num_0_simplices_k,
        *num_1_simplices_k,
        &boundary_1.csr,
    );
    up_persistent + down_persistent
}

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
            let k = largest_index / 2;
            Python::with_gil(|_py| {
                group.bench_with_input(
                    criterion::BenchmarkId::new("no homology, single persistent laplacian", n),
                    &n,
                    |b, &_n| {
                        b.iter_batched(
                            || (sparse_boundary_maps.clone(), filt_hash.clone()),
                            |(maps, hash)| {
                                let persistent = single_pair_persistent_laplacian(maps, hash, k);
                                criterion::black_box(persistent);
                            },
                            criterion::BatchSize::SmallInput,
                        )
                    },
                );
            });
            Python::with_gil(|_py| {
                group.bench_with_input(
                    criterion::BenchmarkId::new("dense svd, single persistent laplacian", n),
                    &n,
                    |b, &_n| {
                        b.iter_batched(
                            || (sparse_boundary_maps.clone(), filt_hash.clone()),
                            |(maps, hash)| {
                                let persistent = single_pair_persistent_laplacian(maps, hash, k);
                                let homology = compute_homology_from_persistent_laplacian_dense(
                                    &criterion::black_box(persistent),
                                );
                                criterion::black_box(homology);
                            },
                            criterion::BatchSize::SmallInput,
                        )
                    },
                );
            });
            Python::with_gil(|_py| {
                group.bench_with_input(
                    criterion::BenchmarkId::new("lanczos, single persistent laplacian", n),
                    &n,
                    |b, &_n| {
                        b.iter_batched(
                            || (sparse_boundary_maps.clone(), filt_hash.clone()),
                            |(maps, hash)| {
                                let persistent = single_pair_persistent_laplacian(maps, hash, k);
                                let homology =
                                    compute_homology_from_persistent_laplacian_lanczos_crate(
                                        &criterion::black_box(persistent),
                                    );
                                criterion::black_box(homology);
                            },
                            criterion::BatchSize::SmallInput,
                        )
                    },
                );
            });
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
                                10,
                                Some(0.00001),
                                None,
                                None,
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
                            let persistent = single_pair_persistent_laplacian(maps, hash, k);
                            // criterion::black_box(persistent);
                            let eigenvalues = compute_homology_from_persistent_laplacian_scipy(
                                &criterion::black_box(persistent),
                                &scipy_config,
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

fn bench_process_single_pair_only_homology(c: &mut Criterion) {
    let ns = [100, 200, 400, 800];
    let ds = [1];
    for &d in &ds {
        let group_name = format!(
            "Eigenvalue only computation for pair of simplices from {}-spheres",
            d
        );
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(10);
        for &n in &ns {
            let (sparse_boundary_maps, filt_hash) = helpers::boundary_maps_sphere_from_python(n, d);
            let largest_index = *filt_hash.keys().max().unwrap();
            let k = largest_index / 2;

            let persistent = single_pair_persistent_laplacian(sparse_boundary_maps, filt_hash, k);

            group.bench_with_input(criterion::BenchmarkId::new("dense svd", n), &n, |b, &_n| {
                b.iter_batched(
                    || {},
                    |()| {
                        let homology =
                            compute_homology_from_persistent_laplacian_dense(&persistent);
                        criterion::black_box(homology);
                    },
                    criterion::BatchSize::SmallInput,
                )
            });
            group.bench_with_input(criterion::BenchmarkId::new("lanczos", n), &n, |b, &_n| {
                b.iter_batched(
                    || {},
                    |()| {
                        let homology =
                            compute_homology_from_persistent_laplacian_lanczos_crate(&persistent);
                        criterion::black_box(homology);
                    },
                    criterion::BatchSize::SmallInput,
                )
            });
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
                            return scipy_config;
                        },
                        |scipy_config| {
                            let eigenvalues = compute_homology_from_persistent_laplacian_scipy(
                                &persistent,
                                &scipy_config,
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

criterion_group!(
    benches,
    bench_process_single_pair,
    bench_process_single_pair_only_homology
);
criterion_main!(benches);
