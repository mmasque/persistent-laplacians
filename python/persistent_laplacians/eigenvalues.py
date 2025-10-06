# Compute smallest eigenvalues of filtration
from persistent_laplacians.filtrations import get_filtration_data
from persistent_laplacians import persistent_laplacians as pl
import numpy as np

def compute_eigenvalues(
    data, zero_tol=1e-6, num_indices=None, use_scipy=False, use_stepwise_schur=False, num_nonzero_eigenvalues=1, split_up_down=False, max_dim=None, max_alpha_square=np.inf
):
    boundary_matrices, boundary_maps_index_dict, subsampled_filtration_indices = (
        get_filtration_data(data, num_indices=num_indices, max_alpha_square=max_alpha_square)
    )
    return pl.smallest_eigenvalues(
        boundary_matrices,
        boundary_maps_index_dict,
        zero_tol=zero_tol,
        filtration_subsampling=subsampled_filtration_indices,
        use_scipy=use_scipy,
        use_stepwise_schur=use_stepwise_schur,
        num_nonzero_eigenvalues=num_nonzero_eigenvalues,
        split_up_down=split_up_down,
        max_dim=max_dim,
    )
