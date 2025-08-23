import numpy as np
import gudhi
from bisect import bisect_right
from collections import defaultdict

from persistent_laplacians.filtrations import get_subsampled_filtration_indices


def get_filtration_data(data, num_indices=None):
    cc = gudhi.CubicalComplex(top_dimensional_cells=data)
    cells = cc.all_cells()
    boundary_matrices = compute_boundary_matrices(cells)
    boundary_maps_index_dict = get_boundary_maps_index_dict(cells)
    subsampled_filtration_indices = get_subsampled_filtration_indices(
        sorted(np.unique(cells)), num_indices
    )
    return boundary_matrices, boundary_maps_index_dict, subsampled_filtration_indices


def compute_cell_dimension(coord):
    """
    Compute the dimension of a cell in a GUDHI bitmap grid.
    Dimension is the number of odd coordinates in the cell's grid index.

    Args:
        coord (tuple of int): coordinates in the bitmap grid (multi-dimensional index).

    Returns:
        int: the cell dimension (number of odd entries in coord).
    """
    return sum(c % 2 for c in coord)


def cell_faces(coord):
    """
    Compute the boundary face coordinates of a cell in a cubical grid, without orientation.

    Args:
        coord (tuple of int): the cell's index in the bitmap grid.

    Returns:
        List[tuple]: list of face coordinate tuples.
    """
    mask = tuple(c % 2 for c in coord)
    faces = []
    for axis, bit in enumerate(mask):
        if not bit:
            continue
        for delta in (-1, +1):
            face = list(coord)
            face[axis] += delta
            faces.append(tuple(face))
    return faces


def cell_face_orientation_old(coord, face_coord):
    """
    Compute the orientation (+1 or -1) of a given face in its parent cell.

    Args:
        coord (tuple of int): the parent cell's index in the bitmap grid.
        face_coord (tuple of int): the face's index in the bitmap grid.

    Returns:
        int: orientation sign.
    """
    dim = compute_cell_dimension(coord)
    # find axis along which they differ
    diffs = [fc - c for c, fc in zip(coord, face_coord)]
    axis = next(i for i, d in enumerate(diffs) if d != 0)
    delta = diffs[axis]
    if dim == 1:
        # For edges, the orientation of the end is 1, and of the start -1
        return delta
    else:
        return delta * (-1) ** axis


def cell_face_orientation(coord, face_coord):
    """
    Compute the orientation (+1 or -1) of a given face in its parent cell.
    See Computational Homology by Kaczynski et al., Proposition 2.36.
    Uses cubical boundary rule: orientation = delta * (-1)^{sum(mask[:axis])},
    where mask indicates odd coords and delta=+1 upper, -1 lower.
    """
    mask = tuple(c % 2 for c in coord)
    diffs = [fc - c for c, fc in zip(coord, face_coord)]
    axis = next(i for i, d in enumerate(diffs) if d != 0)
    delta = diffs[axis]
    parity = sum(mask[:axis])
    return delta * ((-1) ** parity)


def cell_boundary_indices(coord):
    """
    Given a multi-dimensional cell index `coord`,
    compute its boundary faces with orientation in an R-valued cubical complex,
    using `cell_faces` to list faces and then determining orientation.

    Uses the sign convention:
      orientation = +1 for the lower face, -1 for the upper face,
      multiplied by (-1)^axis where axis is the differing index.

    Args:
        coord (tuple of int): the cell's index in the bitmap grid.

    Returns:
        List[Tuple[tuple, int]]: list of (face_coord, orientation) pairs.
    """
    faces = cell_faces(coord)
    return [(f, cell_face_orientation(coord, f)) for f in faces]


def get_sorted_cell_indices(cell_filtration):
    """
    Given a cell filtration (a numpy array of all cells in bitmap format),
    returns a dictionary of dimension: dictionary,
    where each dictionary maps cell indices to their index in the sorted list of cells of that dimension.
    """
    dim_cells = {}
    for idx, val in np.ndenumerate(cell_filtration):
        d = compute_cell_dimension(idx)
        if d not in dim_cells:
            dim_cells[d] = []
        dim_cells[d].append((val, idx))

    flat_index = {}
    for d, cells in dim_cells.items():
        sorted_cells = sorted(cells)
        flat_index[d] = {idx: i for i, (_, idx) in enumerate(sorted_cells)}
    return flat_index


def get_boundary_maps_index_dict(cell_filtration):
    """
    Given `cell_filtration` (an ndarray of global filtration ranks, e.g. from map_values_to_indices),
    returns a dict
      { f : { d : count_of_d_cells_with_filtration <= f } for d=0..max_dim }
    """
    index_filtration = map_values_to_indices(cell_filtration)
    max_dim = len(index_filtration.shape)

    dim_to_steps = defaultdict(list)
    for coord, f in np.ndenumerate(index_filtration):
        d = compute_cell_dimension(coord)
        dim_to_steps[d].append(f)
    for d in dim_to_steps:
        dim_to_steps[d].sort()

    all_steps = sorted(set(index_filtration.flat))

    boundary_maps_index_dict = {}
    for f in all_steps:
        counts = {}
        for d in range(max_dim + 1):
            steps = dim_to_steps.get(d, [])
            # bisect_right gives number ≤ f
            counts[d] = bisect_right(steps, f)
        boundary_maps_index_dict[f] = counts

    return boundary_maps_index_dict


def initialize_boundary_matrices(sorted_cell_indices):
    """
    Initialize boundary matrices for each dimension in a cubical complex.
    """
    boundary_matrices = {}
    # max dimension
    max_dim = max(sorted_cell_indices.keys())
    for d in range(1, max_dim + 1):
        # number of cells in this dimension
        num_cells = len(sorted_cell_indices[d])
        # number of cells in lower dimension
        num_cells_lower = len(sorted_cell_indices[d - 1])
        # create a sparse matrix for the boundary
        boundary_matrices[d] = {
            "n_rows": num_cells_lower,
            "n_cols": num_cells,
            "data": np.array([], dtype=float),
            "rows": np.array([], dtype=int),
            "cols": np.array([], dtype=int),
        }
    return boundary_matrices


def map_values_to_indices(arr):
    """
    Replace each float in `arr` with its index in the sorted list of unique values.

    Example:
        arr = np.array([0.1, 0.5, 0.4, 0.4])
        map_values_to_indices(arr)  # returns array([0, 2, 1, 1])
    """
    # np.unique returns sorted unique values and, with return_inverse,
    # the index of each element in the unique array
    _, inv = np.unique(arr, return_inverse=True)
    return inv.reshape(arr.shape)


def compute_boundary_matrices(cells):
    sorted_cell_indices = get_sorted_cell_indices(cells)
    boundary_matrices = initialize_boundary_matrices(sorted_cell_indices)

    # compute the boundary of each cell and update the relevant boundary matrix
    for index, _value in np.ndenumerate(cells):
        cell_dimension = compute_cell_dimension(index)
        if cell_dimension == 0:
            continue
        flat_cell_index = sorted_cell_indices[cell_dimension][index]
        # Get the boundary faces and their orientations
        cell_boundaries = cell_boundary_indices(index)

        for face, orientation in cell_boundaries:
            face_dimension = compute_cell_dimension(face)
            flat_face_index = sorted_cell_indices[face_dimension][face]
            boundary_matrices[cell_dimension]["data"] = np.append(
                boundary_matrices[cell_dimension]["data"], orientation
            )
            boundary_matrices[cell_dimension]["rows"] = np.append(
                boundary_matrices[cell_dimension]["rows"], flat_face_index
            )
            boundary_matrices[cell_dimension]["cols"] = np.append(
                boundary_matrices[cell_dimension]["cols"], flat_cell_index
            )

    return boundary_matrices


import unittest


class TestCubicalFunctions(unittest.TestCase):

    def test_compute_cell_dimension(self):
        self.assertEqual(compute_cell_dimension((0, 0, 0)), 0)
        self.assertEqual(compute_cell_dimension((1, 0, 0)), 1)
        self.assertEqual(compute_cell_dimension((1, 1, 0)), 2)

    def test_get_sorted_cell_indices(self):
        arr = np.array([0.2, 0.1, 0.3, 0.11, 0.21])
        flat = get_sorted_cell_indices(arr)
        self.assertDictEqual(
            flat, {0: {(0,): 0, (2,): 2, (4,): 1}, 1: {(1,): 0, (3,): 1}}
        )

    def test_get_boundary_maps_index_dict(self):
        arr = np.array([0.2, 0.1, 0.3, 0.11, 0.21])
        glob = get_boundary_maps_index_dict(arr)
        self.assertDictEqual(
            glob,
            {
                np.int64(0): {0: 0, 1: 1},
                np.int64(1): {0: 0, 1: 2},
                np.int64(2): {0: 1, 1: 2},
                np.int64(3): {0: 2, 1: 2},
                np.int64(4): {0: 3, 1: 2},
            },
        )


if __name__ == "__main__":
    unittest.main()
