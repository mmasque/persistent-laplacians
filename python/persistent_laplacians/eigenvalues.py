# Compute smallest eigenvalues of filtration
import gudhi as gd
from persistent_laplacians.filtrations import simplices_by_dimension, compute_boundary_matrices, filtration_hash_map
from persistent_laplacians import persistent_laplacians as pl
import numpy as np 

def _sample_n_values(arr, n):
    arr = np.array(arr)
    indices = np.linspace(0, len(arr) - 1, n, dtype=int)
    return arr[indices]

def get_filtration_data(data, num_indices=None):
    alpha = gd.AlphaComplex(points=data)
    st = alpha.create_simplex_tree()
    filtration = list(st.get_filtration())
    unique_filtration_values = sorted(list(set([f for (_, f) in filtration])))
    simplices_by_dim, simplices_by_dim_only_filt = simplices_by_dimension(filtration)
    boundary_matrices = compute_boundary_matrices(simplices_by_dim)
    boundary_maps_index_dict = filtration_hash_map(filtration, simplices_by_dim_only_filt)
    if num_indices is None:
        num_indices = len(unique_filtration_values) 
    subsampled_filtration_indices = _sample_n_values(range(len(unique_filtration_values)), num_indices)
    return boundary_matrices, boundary_maps_index_dict, subsampled_filtration_indices

def compute_eigenvalues(data, zero_tol=1e-6, num_indices=None, use_scipy=False, use_stepwise_schur=False):
    boundary_matrices, boundary_maps_index_dict, subsampled_filtration_indices = get_filtration_data(data, num_indices=num_indices)
    return pl.smallest_eigenvalue(
        boundary_matrices,
        boundary_maps_index_dict,
        zero_tol=zero_tol,
        filtration_subsampling=subsampled_filtration_indices,
        use_scipy=use_scipy,
        use_stepwise_schur=use_stepwise_schur,
    )


