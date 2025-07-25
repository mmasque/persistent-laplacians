use criterion::{criterion_group, criterion_main, Criterion};
use persistent_laplacians::homology::{
    compute_homology_from_persistent_laplacian_dense,
    compute_homology_from_persistent_laplacian_dense_eigen,
    compute_homology_from_persistent_laplacian_lanczos_crate,
    compute_homology_from_persistent_laplacian_scipy, count_nnz_persistent_laplacian,
    ScipyEigshConfig,
};

use persistent_laplacians::persistent_homology_of_filtration;
use pyo3::{
    types::{PyList, PyModule},
    Python,
};

use crate::helpers::TOL;
mod helpers;

fn bench_process_tda(c: &mut Criterion) {
    let ns = [10, 20, 30, 40, 50];
    for &d in &[1, 2] {
        let group_name = format!("Persistent laplacian computation for {}-spheres", d);
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(10);

        for &n in &ns {
            let (sparse_boundary_maps, filt_hash) = helpers::boundary_maps_sphere_from_python(n, d);

            group.bench_with_input(
                criterion::BenchmarkId::new("persistent laplacian: no homology", n),
                &n,
                |b, &_n| {
                    b.iter_batched(
                        || (sparse_boundary_maps.clone(), filt_hash.clone()),
                        |(maps, hash)| {
                            let eigenvalues = persistent_homology_of_filtration(
                                maps,
                                hash,
                                count_nnz_persistent_laplacian,
                                TOL,
                            );
                            criterion::black_box(eigenvalues);
                        },
                        criterion::BatchSize::SmallInput,
                    )
                },
            );

            group.bench_with_input(criterion::BenchmarkId::new("dense svd", n), &n, |b, &_n| {
                b.iter_batched(
                    || (sparse_boundary_maps.clone(), filt_hash.clone()),
                    |(maps, hash)| {
                        let eigenvalues = persistent_homology_of_filtration(
                            maps,
                            hash,
                            compute_homology_from_persistent_laplacian_dense,
                            TOL,
                        );
                        criterion::black_box(eigenvalues);
                    },
                    criterion::BatchSize::SmallInput,
                )
            });

            group.bench_with_input(
                criterion::BenchmarkId::new("dense eigen", n),
                &n,
                |b, &_n| {
                    b.iter_batched(
                        || (sparse_boundary_maps.clone(), filt_hash.clone()),
                        |(maps, hash)| {
                            let eigenvalues = persistent_homology_of_filtration(
                                maps,
                                hash,
                                compute_homology_from_persistent_laplacian_dense_eigen,
                                TOL,
                            );
                            criterion::black_box(eigenvalues);
                        },
                        criterion::BatchSize::SmallInput,
                    )
                },
            );

            // group.bench_with_input(
            //     criterion::BenchmarkId::new("dense lanczos: eigenvalues crate", n),
            //     &n,
            //     |b, &_n| {
            //         b.iter_batched(
            //             || (sparse_boundary_maps.clone(), filt_hash.clone()),
            //             |(maps, hash)| {
            //                 let eigenvalues = persistent_homology_of_filtration(
            //                     maps,
            //                     hash,
            //                     compute_homology_from_persistent_laplacian_eigenvalues,
            //                 );
            //                 criterion::black_box(eigenvalues);
            //             },
            //             criterion::BatchSize::SmallInput,
            //         )
            //     },
            // );

            group.bench_with_input(
                criterion::BenchmarkId::new("persistent laplacian: lanczos crate", n),
                &n,
                |b, &_n| {
                    b.iter_batched(
                        || (sparse_boundary_maps.clone(), filt_hash.clone()),
                        |(maps, hash)| {
                            let eigenvalues = persistent_homology_of_filtration(
                                maps,
                                hash,
                                compute_homology_from_persistent_laplacian_lanczos_crate,
                                TOL,
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
                            let scipy_config =
                                ScipyEigshConfig::new_from_num_nonzero_eigenvalues_tol(2, TOL, py);
                            return (
                                sparse_boundary_maps.clone(),
                                filt_hash.clone(),
                                scipy_config,
                            );
                        },
                        |(maps, hash, scipy_config)| {
                            let eigenvalues = persistent_homology_of_filtration(
                                maps,
                                hash,
                                |p, _zero_tol| {
                                    compute_homology_from_persistent_laplacian_scipy(
                                        p,
                                        &scipy_config,
                                        TOL,
                                    )
                                },
                                TOL,
                            );
                            criterion::black_box(eigenvalues);
                        },
                        criterion::BatchSize::SmallInput,
                    )
                });
            });
        }

        group.finish();
    }
}

criterion_group!(benches, bench_process_tda);
criterion_main!(benches);
