import warnings
import numpy as np
import scipy.sparse as scisparse

from scipy.optimize import linear_sum_assignment
from manifold.spectral_embedding import spectral_embedding
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph


def distance_symmetric(
        points,
        func,
        param,
        **kwargs
):
    # Distances not symmetric
    distances = func(
        points,
        param,
        mode='distance',
        **kwargs,
    )
    # Connectivity not symmetric
    connectivity = func(
        points,
        param,
        mode='connectivity',
        **kwargs,
    )
    # A = (D + D.T) / (C + C.T)
    distances = distances + distances.T
    connectivity = connectivity + connectivity.T
    adjacency = scisparse.csr_matrix((
        distances.data / connectivity.data,
        distances.nonzero()
    ))
    return adjacency


def adjacency_matrix(
        points,
        mode="distance_inverse",
        method="knn",
        param=10,
        **kwargs
):
    """

    :param points
    :param mode:
    :param method:
    :param param: is n_neighbours when method="knn" and radius when method="radius"
    :return: adjacency matrix from pointcloud
    """
    if "include_self" in kwargs.keys():
        warnings.warn("Ignoring include_self argument in compute_adjacency_matrix, set to False.")
        kwargs.pop("include_self")
    # TODO: raise error similar to other parts
    if method == "knn":
        func = kneighbors_graph
    elif method == "radius":
        func = radius_neighbors_graph
    else:
        raise ValueError("Method " + method + " non supported for adjacency matrix computation.")
    # Note: these adjacency matrices are not symmetric
    if mode == "connectivity" or mode == "distance":
        return func(
            points,
            param,
            mode=mode,
            **kwargs,
        )
    if mode == "distance_symmetric":
        return distance_symmetric(points, func, param, **kwargs)
    # If i,j are connected then A_{i,j} = 1/d_{i,j}, is symmetric
    if mode == "distance_inverse":
        adjacency = distance_symmetric(points, func, param, **kwargs)
        # Perform 1/d_{i,j} accessing data of sparse matrix
        np.reciprocal(adjacency.data, out=adjacency.data)
        return adjacency
    if mode == "gaussian":
        adjacency = distance_symmetric(points, func, param, **kwargs)
        np.exp(-adjacency.data ** 2 / np.max(adjacency.data) ** 2, out=adjacency.data)
        return adjacency
    else:
        raise ValueError("Mode " + mode + " non supported for adjacency matrix computation.")


def length_from_fiedler(
        points,
        threshold=100
):
    """
    Compute the length of a point cloud from its Fiedler vector
    :param points:
    :param threshold:
    :return:
    """
    adjacency = adjacency_matrix(points, mode="gaussian", method="knn")
    _, dd = csgraph_laplacian(
        adjacency, normed=False, return_diag=True
    )

    fiedler = spectral_embedding(
        adjacency=adjacency,
        n_components=1,
        eigen_solver='amg',
        eigen_tol='auto',
        norm_laplacian=False,
        drop_first=True,
        B=scisparse.diags(1 / dd)
    ).reshape(-1)
    side1 = np.mean(points[np.argsort(fiedler)[:threshold]], axis=0)
    side2 = np.mean(points[np.argsort(-fiedler)[:threshold]], axis=0)
    return np.linalg.norm(side2 - side1)


def permutation_matrix(a, b):
    # Check a, b same size
    permutation = np.zeros((len(a), len(a)))
    permutation[a, b] = 1
    return permutation


def bhattacharyya_distance(hist1, hist2):
    """
    Calculate the Bhattacharyya distance between two histograms.

    Args:
    hist1 (numpy.ndarray): The first histogram.
    hist2 (numpy.ndarray): The second histogram.

    Returns:
    float: Bhattacharyya distance between the two histograms.
    """
    if len(hist1) != len(hist2):
        raise ValueError("Histograms must have the same number of bins.")

    # Normalize the histograms
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    # Calculate Bhattacharyya coefficient
    b_coeff = np.sum(np.sqrt(hist1 * hist2))

    # Calculate Bhattacharyya distance
    b_distance = -np.log(b_coeff)

    return b_distance


def compare_signatures(
    eigenmaps_a,
    eigenmaps_b,
    bins=200
):
    assert eigenmaps_a.shape[1] == eigenmaps_b.shape[1], "Eigenmaps should have same size"

    dissimilarities = np.ones((eigenmaps_a.shape[1], eigenmaps_a.shape[1]))
    sign = np.zeros((eigenmaps_a.shape[1], eigenmaps_a.shape[1])) * np.nan
    for ind_a, eigenmap_a in enumerate(eigenmaps_a.T):
        for ind_b, eigenmap_b in enumerate(eigenmaps_b.T):
            hist_a, bin_edges_a = np.histogram(eigenmap_a, bins)
            hist_b, bin_edges_b = np.histogram(eigenmap_b, bins)
            hist_b_neg, bin_edges_b_neg = np.histogram(-eigenmap_b, bins)
            # cosine dissimilarity
            dis = bhattacharyya_distance(hist_a, hist_b)
            dis_inv = bhattacharyya_distance(hist_a, hist_b_neg)
            dissimilarities[ind_a, ind_b] = min(dis, dis_inv)
            sign[ind_a, ind_b] = 1 if dis < dis_inv else -1
    hungarian = linear_sum_assignment(dissimilarities)
    permutation = permutation_matrix(*hungarian) * sign
    transform = permutation @ eigenmaps_b.T
    return transform.T
