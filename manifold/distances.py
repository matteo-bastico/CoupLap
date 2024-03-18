import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import directed_hausdorff


def chamfer_distance(pc_a, pc_b):
    pw_dist = pairwise_distances(pc_a, pc_b)
    return np.mean(np.min(pw_dist, axis=0)) + np.mean(np.min(pw_dist, axis=1))


def hausdorff_distance(pc_a, pc_b):
    # Calculate the directed Hausdorff distance from point_cloud1 to point_cloud2
    directed_distance = directed_hausdorff(pc_a, pc_b)
    # Calculate the directed Hausdorff distance from point_cloud2 to point_cloud1
    directed_distance2 = directed_hausdorff(pc_b, pc_a)
    # To obtain the undirected Hausdorff distance, take the maximum of the directed distances
    undirected_distance = max(directed_distance, directed_distance2)
    return undirected_distance


def grassmann_distance(A, B):
    """
    Compute the Grassmann distance between subspaces spanned by columns of A and B.

    Parameters:
    - A, B: matrices whose columns span the subspaces.
    """

    A_norm = A / np.linalg.norm(A, axis=0)
    B_norm = B / np.linalg.norm(B, axis=0)

    # Orthonormalize the basis for each subspace
    Q_A, _ = np.linalg.qr(A_norm)
    Q_B, _ = np.linalg.qr(B_norm)

    # Compute the matrix of inner products
    C = Q_A.T @ Q_B
    # Get the singular values of C
    singular_values = np.linalg.svd(C, compute_uv=False)

    # The singular values are the cosine of the principal angles.
    # So, get the sine values by taking the square root of (1 - singular_value^2)
    sin_principal_angles = np.sqrt(1 - singular_values ** 2)

    # Grassmann distance is the Frobenius norm of the sine values
    distance = np.linalg.norm(sin_principal_angles)

    return distance