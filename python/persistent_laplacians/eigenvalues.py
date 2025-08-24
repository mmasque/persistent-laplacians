# Compute smallest eigenvalues of filtration
from persistent_laplacians.filtrations import get_filtration_data
from persistent_laplacians import persistent_laplacians as pl


def compute_eigenvalues(
    data, zero_tol=1e-6, num_indices=None, use_scipy=False, use_stepwise_schur=False
):
    boundary_matrices, boundary_maps_index_dict, subsampled_filtration_indices = (
        get_filtration_data(data, num_indices=num_indices)
    )
    return pl.smallest_eigenvalue(
        boundary_matrices,
        boundary_maps_index_dict,
        zero_tol=zero_tol,
        filtration_subsampling=subsampled_filtration_indices,
        use_scipy=use_scipy,
        use_stepwise_schur=use_stepwise_schur,
    )
