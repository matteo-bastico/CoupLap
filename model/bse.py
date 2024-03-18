import numpy as np
import open3d as o3d
import scipy.sparse as sp

from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

from utils.pre_processing import ransac_registration
from manifold.coupling import couple_adjacency_matrices
from manifold.spectral_embedding import spectral_embedding
from manifold.utils import adjacency_matrix, length_from_fiedler
from manifold.distances import chamfer_distance, hausdorff_distance, grassmann_distance

_METHODS = [
    'spectral',
    'chamfer',
    'hausdorff',
    'FPFH',
]
_SEED = 42


class BoneSideEstimation:
    def __init__(
            self,
            source: np.array,
            method: str = 'spectral',
            voxel_size: int = 2,
            cross: float = 0.5,
            maps: int = 20
    ):
        """

        :param source:
        :param method:
        """
        self.source = source
        if method not in _METHODS:
            raise ValueError(f"'mode' must be one of {_METHODS}, received {repr(method)}.")
        self.method = method
        self.voxel_size = voxel_size
        self.cross = cross
        self.maps = maps
        # Pre-process source
        # Source length
        self.l_source = length_from_fiedler(source, threshold=100)
        # Mirroring source
        pca = PCA(n_components=3)
        source_pca = pca.fit_transform(self.source)
        source_pca[:, 1] = -source_pca[:, 1]
        self.mirrored_source = pca.inverse_transform(source_pca)

    def predict(
            self,
            target: np.array,
    ):
        # Use Fiedler for scaling in spectral matching
        if self.method == "spectral":
            l_target = length_from_fiedler(target, threshold=100)
            scale_target = self.l_source / l_target
            target = scale_target * target
        # Do RANSAC registrations
        registered_source = ransac_registration(
            source=self.source,
            target=target,
            voxel_size=self.voxel_size,
            verbose=0
        )
        registered_mirrored_source = ransac_registration(
            source=self.mirrored_source,
            target=target,
            voxel_size=self.voxel_size,
            verbose=0
        )
        if self.method == "spectral":
            dist_ori, dist_mirror = self._predict_spectral(target, registered_source, registered_mirrored_source)
        elif self.method == "chamfer":
            dist_ori, dist_mirror = self._predict_chamfer(target, registered_source, registered_mirrored_source)
        elif self.method == "hausdorff":
            dist_ori, dist_mirror = self._predict_hausdorff(target, registered_source, registered_mirrored_source)
        elif self.method == "FPFH":
            dist_ori, dist_mirror = self._predict_fpfh(target, registered_source, registered_mirrored_source)
        # Final decision
        # Return True if same size, False if opposite
        return dist_ori <= dist_mirror

    def _predict_spectral(self,target: np.array,source,mirrored_source):
        cross_target = np.random.choice(
            range(len(target)),
            size=int(self.cross * len(target)),
            replace=False
        )
        cross_kdtree = KDTree(source)
        dist_source, cross_source = cross_kdtree.query(target[cross_target])
        dist_source = np.exp(-dist_source ** 2 / (np.max(dist_source) ** 2 + np.finfo(float).eps))
        cross_kdtree_mirrored = KDTree(mirrored_source)
        dist_mirrored_source, cross_mirrored_source = cross_kdtree_mirrored.query(target[cross_target])
        dist_mirrored_source = np.exp(
            -dist_mirrored_source ** 2 / (np.max(dist_mirrored_source) ** 2 + np.finfo(float).eps))
        adjacency_target = adjacency_matrix(target, mode="gaussian", method="knn")
        _, dd_target = csgraph_laplacian(
            adjacency_target, normed=False, return_diag=True
        )
        adjacency_source = adjacency_matrix(source, mode="gaussian", method="knn")
        _, dd_source = csgraph_laplacian(
            adjacency_source, normed=False, return_diag=True
        )
        adjacency_mirrored_source = adjacency_matrix(mirrored_source, mode="gaussian", method="knn")
        _, dd_mirrored_source = csgraph_laplacian(
            adjacency_mirrored_source, normed=False, return_diag=True
        )
        # Build coupled adjacency matrix
        adjacency_coupled = couple_adjacency_matrices(
            adjacency_a=adjacency_target,
            adjacency_b=adjacency_source,
            inds_a=cross_target.reshape(-1),
            inds_b=cross_source.reshape(-1),
            dist=sp.diags(dist_source.reshape(-1)),
            return_uncoupled=False
        )
        adjacency_coupled = couple_adjacency_matrices(
            adjacency_a=adjacency_coupled,
            adjacency_b=adjacency_mirrored_source,
            inds_a=cross_target.reshape(-1),
            inds_b=cross_mirrored_source.reshape(-1),
            dist=sp.diags(dist_mirrored_source.reshape(-1)),
            return_uncoupled=False
        )
        coupled_embeddings = spectral_embedding(
            adjacency=adjacency_coupled,
            n_components=self.maps,
            eigen_solver='amg',
            random_state=_SEED,
            eigen_tol='auto',
            norm_laplacian=False,
            drop_first=True,
            B=sp.diags(np.hstack(
                (dd_target,
                 dd_source,
                 dd_mirrored_source
                 )
            ))
        )
        # Split them
        target_embeddings = coupled_embeddings[:len(target)]
        source_embeddings = coupled_embeddings[len(target):len(source) + len(target)]
        mirrored_source_embeddings = coupled_embeddings[len(source) + len(target):]
        # Restriction to cross points
        source_subemb = source_embeddings[cross_source.reshape(-1)]
        target_subemb = target_embeddings[cross_target]
        mirrored_source_subemb = mirrored_source_embeddings[cross_mirrored_source.reshape(-1)]
        # Compute Grassman distances
        grassman_normal = grassmann_distance(target_subemb, source_subemb)
        grassman_mirror = grassmann_distance(target_subemb, mirrored_source_subemb)
        return grassman_normal, grassman_mirror

    def _predict_chamfer(
            self,
            target: np.array,
            source,
            mirrored_source
    ):
        chamfer_normal = chamfer_distance(
            source,
            target
        )
        chamfer_mirror = chamfer_distance(
            mirrored_source,
            target
        )
        return chamfer_normal, chamfer_mirror

    def _predict_hausdorff(
            self,
            target: np.array,
            source,
            mirrored_source
    ):
        hausdorff_normal = hausdorff_distance(
            source,
            target
        )
        hausdorff_mirror = hausdorff_distance(
            mirrored_source,
            target
        )
        return hausdorff_normal, hausdorff_mirror

    def _predict_fpfh(
            self,
            target: np.array,
            source,
            mirrored_source
    ):
        cross_target = np.random.choice(
            range(len(target)),
            size=int(self.cross * len(target)),
            replace=False
        )
        cross_kdtree = KDTree(source)
        _, cross_source = cross_kdtree.query(target[cross_target])
        cross_kdtree_mirrored = KDTree(mirrored_source)
        _, cross_mirrored_source = cross_kdtree_mirrored.query(target[cross_target])
        # Distance of FPFH
        fpfh_source = compute_fpfh(source[cross_source])
        fpfh_target = compute_fpfh(target[cross_target])
        fpfh_mirrored_target = compute_fpfh(mirrored_source[cross_mirrored_source])
        fpfh_dist_normal = np.mean(cosine_distances(fpfh_source, fpfh_target).diagonal())
        fpfh_dist_mirror = np.mean(cosine_distances(fpfh_source, fpfh_mirrored_target).diagonal())
        return fpfh_dist_normal, fpfh_dist_mirror


def compute_fpfh(pc, radius_normal=20, radius_feature=50):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return np.asarray(pcd_fpfh.data)
