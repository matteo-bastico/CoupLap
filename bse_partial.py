import time
import pickle
import random
import argparse
import numpy as np
import scipy.sparse as scisparse
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial import KDTree
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
    parser.add_argument('-th', '--threshold', type=int, default=13000,
                        help='Number of points to subsample for matching')
    parser.add_argument("--verbose", action="store_true",
                        help="activate output visual verbosity", default=True)
    args = parser.parse_args()

    with open(args.source, 'rb') as f:
        source = pickle.load(f)

    with open(args.target, 'rb') as f:
        target = pickle.load(f)

    start = time.time()
    # Pre-processing
    '''
    # Scale the target
    l_source = length_from_fiedler(source, threshold=100)
    l_target = length_from_fiedler(target, threshold=100)
    scale_target = l_source / l_target
    scaled_target = scale_target * target
    '''

    # Select sub-set of the pointcloud
    scaled_target = target[np.argsort(target[:, 2])[:3000], :]

    registered_source = ransac_registration(
        source=source,
        target=scaled_target,
        voxel_size=args.voxel_size
    )
    if args.verbose:
        # fig = plot_pointcloud(scaled_target, name="Scaled Target", marker={'color': '#E3DAC9'}, axis=False)
        fig = plot_pointcloud(registered_source, name="Source", marker={'color': '#E3DAC9', 'size':5}, axis=False)
        fig.update_layout(title="Registered Target-Source")
        fig.show()
    '''
    print("Source registration done")
    print("Mirroring source ...")
    # Mirroring source
    pca = PCA(n_components=3)
    source_pca = pca.fit_transform(source)
    source_pca[:, 1] = -source_pca[:, 1]
    mirrored_source = pca.inverse_transform(source_pca)
    # Do RANSAC registration
    registered_mirrored_source = ransac_registration(
        source=mirrored_source,
        target=scaled_target,
        voxel_size=args.voxel_size
    )
    if args.verbose:
        fig = plot_pointcloud(scaled_target, name="Target")
        fig = plot_pointcloud(registered_mirrored_source, fig=fig, name="Mirrored Source")
        fig.update_layout(title="Registered Target-Mirrored Source")
        fig.show()
    print("Mirrored source registration done")
    '''
    cross_target = np.random.choice(
        range(len(scaled_target)),
        size=int(args.cross * len(scaled_target)),
        replace=False
    )
    print(cross_target)
    print("Coupling graphs ...")
    print(":: Number of cross connections ", int(args.cross * len(scaled_target)))
    cross_kdtree = KDTree(registered_source)
    dist_source, cross_source = cross_kdtree.query(scaled_target[cross_target])
    print(scaled_target[cross_target])
    print(dist_source)
    print(cross_source)
    dist_source = np.exp(-dist_source ** 2 / (np.max(dist_source) ** 2 + np.finfo(float).eps))

    if args.verbose:
        # Create a 3D line plot connecting each pair of points
        lines = []
        for point_set1, point_set2 in zip(registered_source[cross_source], scaled_target[cross_target]):
            lines.append(go.Scatter3d(
                x=[point_set1[0], point_set2[0]],
                y=[point_set1[1], point_set2[1]],
                z=[point_set1[2], point_set2[2]],
                mode='lines',
                line=dict(color='green', width=3)
            ))

        fig = plot_pointcloud(scaled_target, name="Scaled Target")
        fig = plot_pointcloud(registered_source, fig=fig, name="Source")
        fig.update_layout(title="Registered Target-Source")
        fig.add_traces(lines)
        fig.show()
    '''
    cross_kdtree_mirrored = KDTree(registered_mirrored_source)
    dist_mirrored_source, cross_mirrored_source = cross_kdtree_mirrored.query(target[cross_target])
    dist_mirrored_source = np.exp(
        -dist_mirrored_source ** 2 / (np.max(dist_mirrored_source) ** 2 + np.finfo(float).eps))
    '''
    adjacency_target = adjacency_matrix(scaled_target, mode="gaussian", method="knn")
    _, dd_target = csgraph_laplacian(
        adjacency_target, normed=False, return_diag=True
    )
    adjacency_source = adjacency_matrix(registered_source, mode="gaussian", method="knn")
    _, dd_source = csgraph_laplacian(
        adjacency_source, normed=False, return_diag=True
    )
    '''
    adjacency_mirrored_source = adjacency_matrix(registered_mirrored_source, mode="gaussian", method="knn")
    _, dd_mirrored_source = csgraph_laplacian(
        adjacency_mirrored_source, normed=False, return_diag=True
    )
    '''
    # Build coupled adjacency matrix
    adjacency_coupled = couple_adjacency_matrices(
        adjacency_a=adjacency_target,
        adjacency_b=adjacency_source,
        inds_a=cross_target.reshape(-1),
        inds_b=cross_source.reshape(-1),
        dist=scisparse.diags(dist_source.reshape(-1)),
        return_uncoupled=False
    )
    '''
    adjacency_coupled = couple_adjacency_matrices(
        adjacency_a=adjacency_coupled,
        adjacency_b=adjacency_mirrored_source,
        inds_a=cross_target.reshape(-1),
        inds_b=cross_mirrored_source.reshape(-1),
        dist=scisparse.diags(dist_mirrored_source.reshape(-1)),
        return_uncoupled=False
    )
    '''
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
             # dd_mirrored_source
             )
        ))
    )
    # Split them
    target_embeddings = coupled_embeddings[:len(scaled_target)]
    source_embeddings = coupled_embeddings[len(scaled_target):len(source) + len(scaled_target)]
    # mirrored_source_embeddings = coupled_embeddings[len(source) + len(target):]
    # Restriction to cross points
    source_subemb = source_embeddings[cross_source.reshape(-1)]
    target_subemb = target_embeddings[cross_target]
    # mirrored_source_subemb = mirrored_source_embeddings[cross_mirrored_source.reshape(-1)]
    # Compute Grassman distances
    grassman_normal = grassmann_distance(target_subemb, source_subemb)
    # grassman_mirror = grassmann_distance(target_subemb, mirrored_source_subemb)
    '''
    if grassman_normal <= grassman_mirror:
        print("RESULT: Bones are from the SAME side")
    else:
        print("RESULT: Bones are from OPPOSITE sides")
    '''
    total_time = time.time() - start
    print("BSE performed in ", time.time() - start, "seconds.")
    # Show first 6 coupled eigenmaps if verbose is True
    if args.verbose:
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]] * 1,
            vertical_spacing=0.0,
            horizontal_spacing=0.00,
            column_titles=['Source', 'Target', 'Mirrored Source'],
            row_titles=['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', 'Mode 5']
        )
        for i in range(4, 5):
            fig = plot_pointcloud(
                scaled_target,
                marker=dict(
                    size=5,
                    color=target_embeddings[:, i],
                    colorscale='jet',
                    cmax=np.max(coupled_embeddings[:, i]),
                    cmin=np.min(coupled_embeddings[:, i])
                ),
                fig=fig,
                col=2,
                row=i + 1 - 4,
                axis=False
            )
            fig = plot_pointcloud(
                registered_source,
                marker=dict(
                    size=5,
                    color=source_embeddings[:, i],
                    colorscale="Jet",
                    cmax=np.max(coupled_embeddings[:, i]),
                    cmin=np.min(coupled_embeddings[:, i])
                ),
                fig=fig,
                col=1,
                row=i + 1 - 4,
                axis=False
            )

            '''
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
            '''
        fig.update_layout(
            plot_bgcolor='White',
            paper_bgcolor='White',
            showlegend=False,

        )

        fig.for_each_annotation(lambda a: a.update(y=-0.2) if a.text in ['Source', 'Target', 'Mirrored Source'] else a.update(
            x=-0.07) if a.text in ['Mode 1', 'Mode 2', 'Mode 3', 'Mode 4', 'Mode 5'] else ())

        fig.show()
