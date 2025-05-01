import tadasets
from persistent_laplacians.filtrations import simplices_by_dimension, compute_boundary_matrices, filtration_hash_map
import gudhi as gd

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