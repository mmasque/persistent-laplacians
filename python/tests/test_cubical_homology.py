from persistent_laplacians.barcodes import compute_barcodes
from persistent_laplacians import persistent_laplacians as pl
from persistent_laplacians.cubical_complexes import get_filtration_data
import gudhi
import numpy as np
import pytest

DIMS = [(1, 1), (1, 1, 1), (10, 10), (3, 3, 3), (2, 2, 2, 2)]


@pytest.mark.parametrize("dims", DIMS)
def test_cubical_homology(dims):
    np.random.seed(42)
    image = np.random.rand(*dims)

    cc = gudhi.CubicalComplex(top_dimensional_cells=image)
    cells = cc.all_cells()
    boundary_matrices, boundary_maps_index_dict, _ = get_filtration_data(image)

    # RUST call via maturin
    result = pl.process_tda(
        boundary_matrices,
        boundary_maps_index_dict,
        1e-12,
    )

    # verification
    unique_filtration_values = np.unique(cells)
    barcodes = compute_barcodes(result, unique_filtration_values)
    barcodes = [
        (
            (q, (unique_filtration_values[i], unique_filtration_values[j]))
            if j != np.inf
            else (q, (unique_filtration_values[i], np.inf))
        )
        for q in barcodes.keys()
        for (i, j) in barcodes[q].keys()
    ]
    sorted(barcodes)
    print(barcodes)

    # TODO: somehow repeats for now in our code, so remove
    gudhi_persistence = [(q, (i, j)) for (q, (i, j)) in cc.persistence()]
    unique_gudhi = sorted(list(set(gudhi_persistence)))
    print(unique_gudhi)
    unique_persistence = sorted(list(set(barcodes)))
    assert (
        unique_gudhi == unique_persistence
    ), f"Mismatch for image with dimensions {dims}"
