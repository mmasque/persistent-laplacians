use criterion::{criterion_group, criterion_main, Criterion};
use persistent_laplacians::eigenvalues::{
    compute_eigenvalues_from_persistent_laplacian_primme_crate, empty,
};
use persistent_laplacians::homology::{eigsh_scipy, ScipyEigshConfig};
use persistent_laplacians::persistent_eigenvalues_of_filtration;
use pyo3::Python;
mod helpers;

fn bench_compute_eigenvalues(c: &mut Criterion) {
    let ns = [10, 20, 30, 40, 50];
    for &d in &[1, 2] {
        let group_name = format!("Persistent laplacian computation for {}-spheres", d);
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(10);

        for &n in &ns {
            let (sparse_boundary_maps, filt_hash) = helpers::boundary_maps_sphere_from_python(n, d);

            group.bench_with_input(
                criterion::BenchmarkId::new("persistent laplacian: no eigenvalues", n),
                &n,
                |b, &_n| {
                    b.iter_batched(
                        || (sparse_boundary_maps.clone(), filt_hash.clone()),
                        |(maps, hash)| {
                            let eigenvalues =
                                persistent_eigenvalues_of_filtration(maps, hash, empty, 1, None);
                            criterion::black_box(eigenvalues);
                        },
                        criterion::BatchSize::SmallInput,
                    )
                },
            );

            // group.bench_with_input(
            //     criterion::BenchmarkId::new("persistent laplacian: primme", n),
            //     &n,
            //     |b, &_n| {
            //         b.iter_batched(
            //             || (sparse_boundary_maps.clone(), filt_hash.clone()),
            //             |(maps, hash)| {
            //                 let eigenvalues = persistent_eigenvalues_of_filtration(
            //                     maps,
            //                     hash,
            //                     compute_eigenvalues_from_persistent_laplacian_primme_crate,
            //                     1,
            //                     None
            //                 );
            //                 criterion::black_box(eigenvalues);
            //             },
            //             criterion::BatchSize::SmallInput,
            //         )
            //     },
            // );
            Python::with_gil(|py| {
                group.bench_with_input(
                    criterion::BenchmarkId::new("persistent laplacian: scipy", n),
                    &n,
                    |b, &_n| {
                        b.iter_batched(
                            || {
                                (
                                    sparse_boundary_maps.clone(),
                                    filt_hash.clone(),
                                    ScipyEigshConfig::default_from_num_nonzero_eigenvalues(1, py),
                                )
                            },
                            |(maps, hash, config)| {
                                let eigenvalues = persistent_eigenvalues_of_filtration(
                                    maps,
                                    hash,
                                    |matrix, _num_nonzero| {
                                        eigsh_scipy(matrix, &config).unwrap_or(vec![])
                                    },
                                    1,
                                    None,
                                );
                                criterion::black_box(eigenvalues);
                            },
                            criterion::BatchSize::SmallInput,
                        )
                    },
                );
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_compute_eigenvalues);
criterion_main!(benches);
