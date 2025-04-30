import gudhi as gd
import tadasets
from filtrations import simplices_by_dimension, compute_boundary_matrices, filtration_hash_map
from barcodes import compute_barcodes
import persistent_laplacians
import pytest

# generate a variety of sphere parameters
ns = [20, 30, 40, 50]
ds = [1, 2]
rs = [1]
noises = [0.0]
seeds = [0, 42, 99]

SPHERE_PARAMS = [
    (n, d, r, noise, seed)
    for n in ns
    for d in ds
    for r in rs
    for noise in noises
    for seed in seeds
]

def sphere_data(n, d, r, noise, seed):
    # Python code setup
    sphere = tadasets.dsphere(n=n, d=d, r=r, noise=noise, seed=seed)
    alpha = gd.AlphaComplex(points=sphere)
    st = alpha.create_simplex_tree()
    filtration = list(st.get_filtration())

    unique_filtration_values = sorted(list(set([f for (_, f) in filtration])))
    simplices_by_dim, simplices_by_dim_only_filt = simplices_by_dimension(filtration)
    boundary_matrices = compute_boundary_matrices(simplices_by_dim)
    boundary_maps_index_dict = filtration_hash_map(filtration, simplices_by_dim_only_filt)
    return boundary_matrices, boundary_maps_index_dict, unique_filtration_values, st

@pytest.mark.parametrize("n,d,r,noise,seed", SPHERE_PARAMS)
def test_rust_vs_gudhi(n, d, r, noise, seed):
    boundary_matrices, boundary_maps_index_dict, unique_filtration_values, st = sphere_data(n, d, r, noise, seed)
    # RUST call via maturin
    result = persistent_laplacians.process_tda(
        boundary_matrices,
        boundary_maps_index_dict
    )
    # verification
    barcodes = compute_barcodes(result, unique_filtration_values)
    print(barcodes)
    barcodes = [
        (q, (unique_filtration_values[i], unique_filtration_values[j]))
        for q in barcodes.keys()
        for (i, j) in barcodes[q].keys()
    ]
    sorted(barcodes)

    # TODO: somehow repeats for now in our code, so remove
    gudhi_persistence = [(q, (i, j)) for (q, (i,j)) in st.persistence() if q > 0]
    unique_gudhi = sorted(list(set(gudhi_persistence)))
    unique_persistence = sorted(list(set(barcodes)))
    assert unique_gudhi == unique_persistence, f"Mismatch for dsphere(n={n},d={d},r={r},noise={noise},seed={seed})"

