import copy
import numpy as np
import open3d as o3d
import scipy.sparse as sp


def preprocess(
        pc: np.array,
        voxel_size: float
) -> (o3d.geometry.PointCloud, o3d.geometry.PointCloud):
    """
    Down sampling and FPFH for the given pcd
    :param pc
    :param voxel_size
    :return: down sampled and FPFH pointcloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def ransac_registration(
        target: np.array,
        source,
        verbose=True,
        voxel_size=10
):
    """
    Global registration of the full bones, this is done to have a rough initialization for the complete or
    split local registration
    :param target
    :param return_downsampled
    :return:
    """

    source_down, source_fpfh = preprocess(source, voxel_size)
    distance_threshold = voxel_size * 1.5
    if verbose:
        print("RANSAC registration on downsampled point clouds.")
        print(":: Since the downsampling voxel size is %.3f," % voxel_size)
        print(":: we use a liberal distance threshold %.3f." % distance_threshold)
    #
    target_down, target_fpfh = preprocess(target, voxel_size)
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(source)
    return np.asarray(
        copy.deepcopy(pcd).transform(res.transformation).points
    )


def largest_connected_component(
    data,
    adj,
    num_components: int = 1,
    connection: str = 'weak'
):
    n, component = sp.csgraph.connected_components(adj, connection=connection)
    if n <= num_components:
            return data
    _, count = np.unique(component, return_counts=True)
    subset = np.in1d(component, count.argsort()[-num_components:])
    return data[subset, :]
