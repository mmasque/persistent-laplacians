import petls
from persistent_laplacians.eigenvalues import compute_eigenvalues
import tadasets
import math
import pytest
import math
import gudhi as gd


def compare_impls(dataset, use_scipy, use_stepwise_schur):
    alpha_petls = petls.Alpha(points=dataset)

    petls_filtration_values = alpha_petls.get_all_filtrations()

    pl_dict = compute_eigenvalues(
        dataset,
        zero_tol=1e-9,
        use_scipy=use_scipy,
        use_stepwise_schur=use_stepwise_schur,
    )
    for dim, filts in sorted(pl_dict.items()):
        for (i, j), eigs in sorted(filts.items()):
            petls_compute, _ = alpha_petls.eigenpairs(
                dim, petls_filtration_values[i], petls_filtration_values[j]
            )
            sorted(petls_compute)
            petls_nonzero = [x for x in petls_compute if abs(x) > 1e-5]
            for k, eig in enumerate(eigs):
                assert math.isclose(
                    petls_nonzero[k], eig, abs_tol=1e-3
                ), f"Different at {dim}, ({i}, {j}), {petls_nonzero[k]} != {eig}"


# This test is janky: it will fail when playing with the parameters of the point clouds (like toggling noise off).
# I think this is the filtration values sometimes become different. I also don't know petls tolerance, so sometimes
# my implementation has a close to zero eigenvalue while petls' implementation doesn't or vice-versa.
def test_rust_vs_petls():
    sphere_2 = tadasets.dsphere(n=20, d=2, r=1, noise=0.1, seed=0)
    sphere_1 = tadasets.dsphere(n=20, d=1, r=1, noise=0.1, seed=0, ambient=3)
    torus = tadasets.torus(n=20, noise=0.1, seed=0)

    compare_impls(sphere_2, True, True)
    compare_impls(sphere_2, True, False)
    compare_impls(sphere_2, False, True)
    compare_impls(sphere_2, False, False)

    compare_impls(sphere_1, True, True)
    compare_impls(sphere_1, True, False)
    compare_impls(sphere_1, False, True)
    compare_impls(sphere_1, False, False)

    compare_impls(torus, True, True)
    compare_impls(torus, True, False)
    compare_impls(torus, False, True)
    compare_impls(torus, False, False)
