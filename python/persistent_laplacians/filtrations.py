import numpy as np
from collections import defaultdict
import gudhi as gd

import bisect


def get_filtration_data(data, num_indices=None):
    alpha = gd.AlphaComplex(points=data)
    st = alpha.create_simplex_tree()
    filtration = list(st.get_filtration())
    unique_filtration_values = sorted(list(set([f for (_, f) in filtration])))
    simplices_by_dim, simplices_by_dim_only_filt = simplices_by_dimension(filtration)
    boundary_matrices = compute_boundary_matrices(simplices_by_dim)
    boundary_maps_index_dict = filtration_hash_map(
        filtration, simplices_by_dim_only_filt
    )
    if num_indices is None:
        num_indices = len(unique_filtration_values)
    subsampled_filtration_indices = _sample_n_values(
        range(len(unique_filtration_values)), num_indices
    )
    return boundary_matrices, boundary_maps_index_dict, subsampled_filtration_indices


def get_subsampled_filtration_indices(unique_filtration_values, num_indices):
    if num_indices is None:
        num_indices = len(unique_filtration_values)
    return _sample_n_values(range(len(unique_filtration_values)), num_indices)


def _sample_n_values(arr, n):
    arr = np.array(arr)
    indices = np.linspace(0, len(arr) - 1, n, dtype=int)
    return arr[indices]


def simplices_by_dimension(filtration):
    simplices_by_dim = defaultdict(list)
    simplices_by_dim_only_filt = defaultdict(list)
    for simplex, filt_value in filtration:
        dim = len(simplex) - 1
        simplices_by_dim[dim].append(tuple(sorted(simplex)))
        simplices_by_dim_only_filt[dim].append(filt_value)
    return simplices_by_dim, simplices_by_dim_only_filt


# For now, this is slow. Probably want to move this to the Rust component, and pass the simplices directly at some point.
def compute_boundary_matrices(simplices):
    """
    Compute boundary matrices for a simplicial complex.

    Parameters:
    simplices (dict): Dictionary where keys are dimensions (int) and values are lists of tuples representing simplices.

    Returns:
    dict: Dictionary where keys are dimensions and values are NumPy arrays representing boundary matrices.
    """

    # Determine the maximum dimension
    max_dim = max(simplices.keys())

    # Create index mappings for each dimension
    index_maps = {
        dim: {simplex: idx for idx, simplex in enumerate(simplices[dim])}
        for dim in simplices
    }

    # Compute boundary matrices
    boundary_matrices = {}
    for dim in range(1, max_dim + 1):
        higher_simplices = simplices.get(dim, [])
        lower_simplices = simplices.get(dim - 1, [])
        lower_index_map = index_maps.get(dim - 1, {})

        rows = []
        cols = []
        data = []
        # for each of the higher simplices
        for col, simplex in enumerate(higher_simplices):
            # for each of the vertices of the simplex
            for i, _ in enumerate(sorted(simplex)):
                # get face missing this index
                face = tuple(simplex[:i] + simplex[i + 1 :])
                # since this is a simplicial complex, the face is in the lower simplices
                row = lower_index_map.get(face)
                rows.append(row)
                cols.append(col)
                data.append((-1) ** i)

        # create sparse matrix
        n_rows = len(lower_simplices)
        n_cols = len(higher_simplices)
        boundary_matrices[dim] = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "data": np.asarray(data, dtype=np.float64),
            "rows": np.asarray(rows, dtype=np.int64),
            "cols": np.asarray(cols, dtype=np.int64),
        }

    return boundary_matrices


def filtration_hash_map(filtration, simplices_by_dim_only_filt):
    def max_index(lst, b):
        # Find the insertion point for b to the right
        idx = bisect.bisect_right(lst, b)
        return idx

    unique_filtration_values = sorted(list(set([f for (_, f) in filtration])))
    # For each filtration value, get indices for the boundary map at that filtration value
    boundary_maps_index_dict = {
        filt_index: {
            key: max_index(filt_values, filt_value)
            for key, filt_values in simplices_by_dim_only_filt.items()
        }
        for (filt_index, filt_value) in enumerate(unique_filtration_values)
    }
    return boundary_maps_index_dict
