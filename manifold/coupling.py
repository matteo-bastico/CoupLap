import numpy as np
import scipy.sparse as scisparse


def couple_adjacency_matrices(
    adjacency_a,
    adjacency_b,
    inds_a,
    inds_b,
    dist,
    return_uncoupled=False
):
    """
    coupling_distances: matrix F_a x F_b with distances, element i,j is d(F_a[i], F_b[j])
    :returns coupled adjacency matrix, if return_uncoupled=True also the uncoupled version
    """
    # Create adjacency uncoupled form two single adjacency
    # This is
    #  | A_a |  0  |
    #  | 0   | A_b |
    adjacency_uncoupled = scisparse.hstack([
        scisparse.vstack([
            # scisparse.csr_matrix(adjacency_a.shape),
            adjacency_a,
            scisparse.csr_matrix(
                (adjacency_b.shape[0],
                 adjacency_a.shape[1])
            )
        ]),
        scisparse.vstack([
            scisparse.csr_matrix(
                (adjacency_a.shape[0],
                 adjacency_b.shape[1])
            ),
            adjacency_b
            # scisparse.csr_matrix(adjacency_b.shape),
        ])
    ])
    # For faster computation, convert to LIL
    adjacency_coupled = adjacency_uncoupled.tolil()
    # adjacency_coupled = adjacency_uncoupled
    adjacency_coupled[
        np.array(inds_a)[:, None],  # This is unchanged
        np.array(inds_b) + adjacency_a.shape[0]  # This is skipped of shape[0]
    ] = dist
    # Transpose part
    adjacency_coupled[
        np.array(inds_b)[:, None] + adjacency_a.shape[0],
        np.array(inds_a),
    ] = adjacency_coupled[
        np.array(inds_a)[:, None],  # This is unchanged
        np.array(inds_b) + adjacency_a.shape[0]  # This is skipped of len(bone_a_raw)
    ].T
    # Back to CSR
    adjacency_coupled = adjacency_coupled.tocsr()
    if return_uncoupled:
        return adjacency_coupled, adjacency_uncoupled
    else:
        return adjacency_coupled