from persistent_laplacians.data import sphere_data
from persistent_laplacians.barcodes import compute_barcodes
from persistent_laplacians import persistent_laplacians as pl
import pytest
import numpy as np

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

@pytest.mark.parametrize("n,d,r,noise,seed", SPHERE_PARAMS)
def test_rust_vs_gudhi(n, d, r, noise, seed):
    boundary_matrices, boundary_maps_index_dict, unique_filtration_values, st = sphere_data(n, d, r, noise, seed)
    # RUST call via maturin
    result = pl.process_tda(
        boundary_matrices,
        boundary_maps_index_dict,
        1e-6,
    )
    # verification
    barcodes = compute_barcodes(result, unique_filtration_values)
    print(barcodes)
    barcodes = [
        (q, (unique_filtration_values[i], unique_filtration_values[j])) if j != np.inf else (q, (unique_filtration_values[i],np.inf))
        for q in barcodes.keys()
        for (i, j) in barcodes[q].keys()
    ]    
    sorted(barcodes)

    # TODO: somehow repeats for now in our code, so remove
    gudhi_persistence = [(q, (i, j)) for (q, (i,j)) in st.persistence() if q > 0]
    unique_gudhi = sorted(list(set(gudhi_persistence)))
    unique_persistence = sorted(list(set(barcodes)))
    assert unique_gudhi == unique_persistence, f"Mismatch for dsphere(n={n},d={d},r={r},noise={noise},seed={seed})"
