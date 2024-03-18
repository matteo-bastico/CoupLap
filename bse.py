import time
import pickle
import random
import probreg
import argparse
import numpy as np
import scipy.sparse as scisparse
import matplotlib.pyplot as plt

import open3d as o3
from scipy.spatial import KDTree
from probreg import callbacks
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
from utils.graphics import plot_pointcloud
from manifold.distances import grassmann_distance
from utils.pre_processing import ransac_registration
from manifold.coupling import couple_adjacency_matrices
from manifold.spectral_embedding import spectral_embedding
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from manifold.utils import adjacency_matrix, length_from_fiedler

_SEED = 42
random.seed = _SEED
np.random.seed = _SEED


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--cross', type=float, default=0.2,
                        help='Fraction of source nodes used for coupled graph building')
    parser.add_argument('-v', '--voxel_size', type=float, default=2,
                        help='Voxel size for point cloud down-sampling')
    parser.add_argument('-c', '--maps', type=int, default=20,
                        help='Number of coupled maps')
    parser.add_argument('-s', '--source', type=str, default='data/BSE_Human/Femur_L/C25LFE.pkl',
                        help='Path to source bone')
    parser.add_argument('-t', '--target', type=str, default='data/BSE_Human/Femur_R/C04RFE.pkl',
                        help='Path to target')
    parser.add_argument('-r', '--registration', type=str, default='GMM+CPD',
                        help='RANSAC or GMM+CPD')
    parser.add_argument('-th', '--threshold', type=int, default=2000,
                        help='Number of points to subsample for matching')
    parser.add_argument("--verbose", action="store_true",
                        help="activate output visual verbosity", default=True)
    parser.add_argument("--use_cuda", action="store_true",
                        help="cuda for CPD", default=True)
    args = parser.parse_args()

    if args.use_cuda:
        import cupy as cp
        to_cpu = cp.asnumpy
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

    with open(args.source, 'rb') as f:
        source = pickle.load(f)
    if source.shape[0] > args.threshold:
        random_points = np.random.choice(range(len(source)), args.threshold, replace=False)
        source = source[random_points]
    with open(args.target, 'rb') as f:
        target = pickle.load(f)
    if target.shape[0] > args.threshold:
        random_points = np.random.choice(range(len(target)), args.threshold, replace=False)
        target = target[random_points]

    start = time.time()
    # Pre-processing
    '''
    # Scale the target
    l_source = length_from_fiedler(source, threshold=100)
    l_target = length_from_fiedler(target, threshold=100)
    scale_target = l_source / l_target
    scaled_target = scale_target * target
    print("Scaling target bone of", scale_target)
    '''
    scaled_target = target
    if args.verbose:
        fig = plot_pointcloud(scaled_target, name="Target")
        fig = plot_pointcloud(source, fig=fig, name="Source")
        fig.update_layout(title="Original Target-Source")
        fig.show()

    if args.registration == 'RANSAC':
        registered_source = ransac_registration(
            source=source,
            target=scaled_target,
            voxel_size=args.voxel_size
        )
    elif args.registration == 'GMM+CPD':
        # GMM
        (transformation, q) = probreg.gmmtree.registration_gmmtree(
            source=source,
            target=scaled_target,
            maxiter=20,
            tf_init_params={
                't': np.mean(source, axis=0)-np.mean(scaled_target, axis=0)
            }
        )
        print(q)
        print(transformation.rot)
        print(transformation.t)
        print(transformation.scale)
        registered_source = transformation.transform(source)

        intermediate_source = registered_source

        # CPD
        if args.use_cuda:
            registered_source = cp.asarray(registered_source)
            scaled_target = cp.asarray(scaled_target)
        (transformation, sigma2, q) = probreg.cpd.registration_cpd(
            source=registered_source,
            target=scaled_target,
            tf_type_name="nonrigid",
            use_cuda=args.use_cuda
        )
        registered_source = transformation.transform(registered_source)
        if args.use_cuda:
            registered_source = to_cpu(registered_source)
            scaled_target = to_cpu(scaled_target)

    if args.verbose:
        fig = plot_pointcloud(scaled_target, name="Target")
        if args.registration == 'GMM+CPD':
            fig = plot_pointcloud(intermediate_source, fig=fig, name="Intermediate Source")
        fig = plot_pointcloud(registered_source, fig=fig, name="Final Source")
        fig.update_layout(title="Registered Target-Source")
        fig.show()

    print("Source registration done")
    print("Mirroring source ...")
    # Mirroring source
    pca = PCA(n_components=3)
    source_pca = pca.fit_transform(source)
    source_pca[:, 1] = -source_pca[:, 1]
    mirrored_source = pca.inverse_transform(source_pca)
    if args.registration == 'RANSAC':
        # Do RANSAC registration
        registered_mirrored_source = ransac_registration(
            source=mirrored_source,
            target=scaled_target,
            voxel_size=args.voxel_size
        )
    elif args.registration == 'GMM+CPD':
        # GMM
        (transformation, q) = probreg.gmmtree.registration_gmmtree(
            source=mirrored_source,
            target=scaled_target,
            maxiter=20,
            tf_init_params={
                't': np.mean(mirrored_source, axis=0) - np.mean(scaled_target, axis=0)
            }
        )
        print(q)
        print(transformation.rot)
        print(transformation.t)
        print(transformation.scale)
        registered_mirrored_source = transformation.transform(mirrored_source)

        intermediate_registered_mirrored_source = registered_mirrored_source
        # CPD
        if args.use_cuda:
            registered_mirrored_source = cp.asarray(registered_mirrored_source)
            scaled_target = cp.asarray(scaled_target)
        (transformation, sigma2, q) = probreg.cpd.registration_cpd(
            source=registered_mirrored_source,
            target=scaled_target,
            tf_type_name="nonrigid",
            use_cuda=args.use_cuda
        )
        registered_mirrored_source = transformation.transform(registered_mirrored_source)
        if args.use_cuda:
            registered_mirrored_source = to_cpu(registered_mirrored_source)
            scaled_target = to_cpu(scaled_target)

    if args.verbose:
        fig = plot_pointcloud(scaled_target, name="Target")
        if args.registration == 'GMM+CPD':
            fig = plot_pointcloud(intermediate_registered_mirrored_source, fig=fig, name="Mirrored Source Intermediate")
        fig = plot_pointcloud(registered_mirrored_source, fig=fig, name="Mirrored Source Final")
        fig.update_layout(title="Registered Target-Mirrored Source")
        fig.show()
    print("Mirrored source registration done")
    cross_target = np.random.choice(
        range(len(target)),
        size=int(args.cross * len(target)),
        replace=False
    )
    print("Coupling graphs ...")
    print(":: Number of cross connections ", int(args.cross * len(target)))
    cross_kdtree = KDTree(registered_source)
    dist_source, cross_source = cross_kdtree.query(target[cross_target])
    dist_source = np.exp(-dist_source ** 2 / (np.max(dist_source) ** 2 + np.finfo(float).eps))
    cross_kdtree_mirrored = KDTree(registered_mirrored_source)
    dist_mirrored_source, cross_mirrored_source = cross_kdtree_mirrored.query(target[cross_target])
    dist_mirrored_source = np.exp(
        -dist_mirrored_source ** 2 / (np.max(dist_mirrored_source) ** 2 + np.finfo(float).eps))
    adjacency_target = adjacency_matrix(target, mode="gaussian", method="knn")
    _, dd_target = csgraph_laplacian(
        adjacency_target, normed=False, return_diag=True
    )
    adjacency_source = adjacency_matrix(registered_source, mode="gaussian", method="knn")
    _, dd_source = csgraph_laplacian(
        adjacency_source, normed=False, return_diag=True
    )
    adjacency_mirrored_source = adjacency_matrix(registered_mirrored_source, mode="gaussian", method="knn")
    _, dd_mirrored_source = csgraph_laplacian(
        adjacency_mirrored_source, normed=False, return_diag=True
    )
    # Build coupled adjacency matrix
    adjacency_coupled = couple_adjacency_matrices(
        adjacency_a=adjacency_target,
        adjacency_b=adjacency_source,
        inds_a=cross_target.reshape(-1),
        inds_b=cross_source.reshape(-1),
        dist=scisparse.diags(dist_source.reshape(-1)),
        return_uncoupled=False
    )
    adjacency_coupled = couple_adjacency_matrices(
        adjacency_a=adjacency_coupled,
        adjacency_b=adjacency_mirrored_source,
        inds_a=cross_target.reshape(-1),
        inds_b=cross_mirrored_source.reshape(-1),
        dist=scisparse.diags(dist_mirrored_source.reshape(-1)),
        return_uncoupled=False
    )
    if args.verbose:
        plt.spy(adjacency_coupled)
        plt.title('Coupled adjacency matrix')
        plt.show()
    print("Compute coupled", args.maps, "eigenmaps")
    coupled_embeddings = spectral_embedding(
        adjacency=adjacency_coupled,
        n_components=args.maps,
        eigen_solver='amg',
        random_state=_SEED,
        eigen_tol='auto',
        norm_laplacian=False,
        drop_first=True,
        B=scisparse.diags(np.hstack(
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

    if grassman_normal <= grassman_mirror:
        print("RESULT: Bones are from the SAME side")
    else:
        print("RESULT: Bones are from OPPOSITE sides")

    total_time = time.time() - start
    print("BSE performed in ", time.time() - start, "seconds.")
    # Show first 6 coupled eigenmaps if verbose is True
    if args.verbose:
        fig = make_subplots(
            rows=5,
            cols=3,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]] * 5,
            vertical_spacing=0.0,
            horizontal_spacing=0.00,
            column_titles=['Source', 'Target', 'Mirrored Source'],
            row_titles=['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', 'Mode 5']
        )
        for i in range(5):
            fig = plot_pointcloud(
                target,
                marker=dict(
                    size=5,
                    color=target_embeddings[:, i],
                    colorscale='jet'
                ),
                fig=fig,
                col=2,
                row=i + 1,
                axis=False
            )
            fig = plot_pointcloud(
                registered_source,
                marker=dict(
                    size=5,
                    color=source_embeddings[:, i],
                    colorscale="Jet"
                ),
                fig=fig,
                col=1,
                row=i + 1,
                axis=None

            )
            fig = plot_pointcloud(
                registered_mirrored_source,
                marker=dict(
                    size=5,
                    color=mirrored_source_embeddings[:, i],
                    colorscale='jet'
                ),
                fig=fig,
                col=3,
                row=i + 1,
                axis=False
            )
        fig.update_layout(
            plot_bgcolor='White',
            paper_bgcolor='White',
            showlegend=False
        )
        '''
        fig.for_each_annotation(lambda a: a.update(y=-0.2) if a.text in ['Source', 'Target', 'Mirrored Source'] else a.update(
            x=-0.07) if a.text in ['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', 'Mode 5'] else ())
        '''
        fig.show()
