use criterion::{criterion_group, criterion_main, Criterion};
use persistent_laplacians::persistent_laplacians_of_filtration;
mod helpers;

fn bench_process_tda(c: &mut Criterion) {
    // the different problem sizes to test
    let ns = [10, 20, 30, 40, 50, 75, 100];

    // loop over the two d‐values we care about
    for &d in &[1, 2] {
        let group_name = format!("tda_d{}", d);
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(10);

        for &n in &ns {
            let (sparse_boundary_maps, filt_hash) = helpers::boundary_maps_sphere_from_python(n, d);

            group.bench_with_input(criterion::BenchmarkId::from_parameter(n), &n, |b, &_n| {
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

        group.finish();
    }
}

criterion_group!(benches, bench_process_tda);
criterion_main!(benches);
