use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use helpers::generate_large_sparse_matrix;
use persistent_laplacians::{
    eigen_persistent_laplacian, persistent_laplacian, up_persistent_laplacian,
};
mod helpers;
fn bench_up_persistent_laplacian(c: &mut Criterion) {
    let sizes = [100, 200, 300, 400, 500];
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Linear);
    let mut group = c.benchmark_group("Up Persistent Laplacian");
    group.plot_config(plot_config);

    for n in sizes {
        let matrix = generate_large_sparse_matrix(n, 2 * n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_s| {
            b.iter(|| {
                let _ = up_persistent_laplacian(&matrix, n - 1);
            });
        });
    }
}

fn bench_persistent_laplacian(c: &mut Criterion) {
    let sizes = [100, 200, 300, 400, 500, 600, 700, 800];
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Linear);
    let mut group = c.benchmark_group("Persistent Laplacian");
    group.plot_config(plot_config);

    for n in sizes {
        let boundary_2 = generate_large_sparse_matrix(n, 3 * n);
        let boundary_1 = generate_large_sparse_matrix(n, 2 * n);
        let dims_qp1_larger = (n, n);
        let dims_qp1_smaller = (n / 2, n / 2); // n/2 q simplices, n/2 q+1 simplices
        let dims_q_smaller = (n / 2, n / 2);
        group.bench_with_input(
            BenchmarkId::new("Eigenvalues of persistent laplacian", n),
            &n,
            |b, &_s| {
                b.iter(|| {
                    let _ = eigen_persistent_laplacian(
                        &boundary_2,
                        dims_qp1_smaller,
                        dims_qp1_larger,
                        &boundary_1,
                        dims_q_smaller,
                    )
                    .unwrap();
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("Persistent Laplacian", n), &n, |b, &_s| {
            b.iter(|| {
                let _ = persistent_laplacian(
                    &boundary_2,
                    dims_qp1_smaller,
                    dims_qp1_larger,
                    &boundary_1,
                    dims_q_smaller,
                )
                .unwrap();
            });
        });
    }
}

criterion_group!(
    benches,
    bench_persistent_laplacian,
    bench_up_persistent_laplacian
);
criterion_main!(benches);
