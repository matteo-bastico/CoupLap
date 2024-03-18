import os
import argparse
import random
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.neighbors import KDTree
from skimage.morphology import disk, closing

from utils.io import load_tiff_pc
from model.ad import AnomalyDetection
from manifold.utils import adjacency_matrix
from utils.pre_processing import largest_connected_component

_SEED = 42
np.random.seed = 42
random.seed = 42


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--cross', type=float, default=1,
                        help='Fraction of source nodes used for coupled graph building')
    parser.add_argument('-c', '--maps', type=int, default=100,
                        help='Number of coupled maps')
    parser.add_argument('-p', '--points', type=int, default=13000,
                        help='Number of points to subsample for matching')
    parser.add_argument('-t', '--threshold', type=float, default=0.001,
                        help='Threshold for plane removal')
    parser.add_argument('-cd', '--cpd_down', type=int, default=3000,
                        help='Number of points to subsample for CPD')
    parser.add_argument('-v', '--voxel_size', type=float, default=0.001,
                        help='RANSAC voxel size')
    parser.add_argument('-cls', '--cls', type=str, default='bagel',
                        help='Folder containing samples')
    parser.add_argument('-s', '--source', type=str, default='data/MVTec 3D-AD/bagel/train/good/xyz/000.tiff',
                        help='Rigid registration type')
    parser.add_argument('-d', '--dataset_dir', type=str, default='data/MVTec 3D-AD',
                        help='Rigid registration type')
    parser.add_argument('-o', '--output_dir', type=str, default='results/MVTec_3D',
                        help='Rigid registration type')
    parser.add_argument('-m', '--method', type=str, default='spectral',
                        help='Matching method')
    parser.add_argument('--bins', type=int, default=200,
                        help='Bins for histogram matching if method==Hist')
    args = parser.parse_args()

    cls = args.cls
    path = os.path.join(args.dataset_dir, cls)
    source_path = args.source

    # Load source point clouds
    source, source_pca, source_sampling = load_tiff_pc(
        source_path,
        n_points=args.points,
        threshold=args.threshold
    )
    # Keep the largest connected components (sometimes background disconnected components may be present)
    adjacency_source = adjacency_matrix(source, mode="gaussian", method="knn")
    source = largest_connected_component(source, adj=adjacency_source, num_components=1)
    # Create the AnomalyDetection object
    ad = AnomalyDetection(
        source=source,
        method=args.method,
        voxel_size=args.voxel_size,
        cross=args.cross,
        maps=args.maps,
        cpd_down=args.cpd_down,
        bins=args.bins
    )
    # List of tasks in the test folder (anomalies list)
    tasks = os.listdir(
        os.path.join(path, 'test')
    )
    results = {}
    # Iterate over tasks
    for task in tasks:
        results[task] = {}
        target_ids = os.listdir(os.path.join(path, 'test', task, 'xyz'))
        # Iterate over test saples -> targets
        for target_id in target_ids:
            print(task, target_id)
            # Do matching and obtain score
            target, target_pca, target_sampling = load_tiff_pc(
                os.path.join(path, 'test', task, 'xyz', target_id),
                n_points=args.points,
                threshold=args.threshold
            )
            # Keep the largest connected components
            adjacency_target = adjacency_matrix(target, mode="gaussian", method="knn")
            target = largest_connected_component(target, adj=adjacency_target, num_components=1)
            # Predict anomalies
            p2p_dist, cross_target = ad.predict(target)
            # Save prediction .tiff file to compare with gt
            # Here we have to find back the cross_target point in the original image
            # and then perform the closing (or dilation)
            img = tiff.imread(os.path.join(path, 'test', task, 'xyz', target_id))
            img_array = np.array(img)
            full_pc = img_array.reshape(-1, 3)
            kdtree = KDTree(full_pc)
            original_points = kdtree.query(target_pca.inverse_transform(target[cross_target]))
            prediction = np.zeros((full_pc.shape[0]))
            prediction[original_points[1].reshape(-1)] = p2p_dist.reshape(-1)
            prediction = prediction.reshape((img_array.shape[0], img_array.shape[1]))
            if target_sampling > 1:
                footprint = disk(int(target_sampling))
                prediction = closing(prediction, footprint)
            '''
            plt.imshow(prediction, interpolation="none")
            plt.title(task + target_id)
            plt.show()
            '''
            # Save file
            im = Image.fromarray(prediction)
            if not os.path.exists(os.path.join(args.output_dir, args.cls, 'test', task)):
                os.makedirs(os.path.join(args.output_dir, args.cls, 'test', task))
            im.save(os.path.join(args.output_dir, args.cls, 'test', task, target_id))
