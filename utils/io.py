import numpy as np
import tifffile as tiff

from scipy.io import loadmat
from sklearn.decomposition import PCA

_FISHER_BONES = [
    'Sacrum',
    'Hip_R',
    'Hip_L',
    'Femur_R',
    'Femur_L'
]


def load_fisher(
        file,
        bone='Femur_L'
):
    """
    Load ONLY bone pointcloud from `.raw` surface models of Fisher et al. (https://zenodo.org/record/4280899)

    :param file:
    :param bone:
    :return:
    """
    if bone not in _FISHER_BONES:
        raise ValueError(f"'error_y_mode' must be one of {_FISHER_BONES}, received {repr(bone)}.")
    mat = loadmat(file)
    for arr in mat['B'][0, :]:
        if arr[0][0] == bone:
            return arr[2][0][0][0]
    raise ValueError(f"Bone {bone} not found in {file}.")


def load_tiff_pc(
        tiff_path,
        n_points: int=None,
        threshold=0.0
):
    """
    Load a point cloud from the MVTec 3D-AD dataset saved as .tiff file.
    Additionally, perform foreground extraction through PCA using the provided threshold
    and downsample it to n_points, if provided.

    :param tiff_path: path to .tiff
    :param n_points: default:None, int type, number of points for random downsampling
    :param threshold: threshold on PCA z-axis for foreground extraction
    :return: pre-processed point cloud, original sklearn.decomposition.PCA and sampling factor
    """
    img = tiff.imread(tiff_path)
    img_array = np.array(img)
    pc = img_array.reshape(-1, 3)
    # Remove zeros (Shadow)
    pc = pc[~np.all(pc == 0, axis=1)]
    pca = PCA(n_components=3)
    pc_pca = pca.fit_transform(pc)
    # Temporal workaround for sign
    # Remove plane (Counting for 3rd component direction)
    pc_no_plane = pc_pca[pc_pca[:, 2] > threshold]
    pc_no_plane2 = pc_pca[pc_pca[:, 2] <= -threshold]
    h1 = np.abs(np.max(pc_no_plane[:, 2]) - np.min(pc_no_plane[:, 2]))
    h2 = np.abs(np.max(pc_no_plane2[:, 2]) - np.min(pc_no_plane2[:, 2]))
    if h2 > h1:
        pc_no_plane2[:, 2] = -pc_no_plane2[:, 2]
        pc_no_plane = pc_no_plane2
    sampling_factor = 1
    if n_points is not None and n_points < len(pc_no_plane):
        sampling_factor = len(pc_no_plane) / n_points
        random_points = np.random.choice(range(len(pc_no_plane)), n_points, replace=False)
        pc_no_plane = pc_no_plane[random_points]
    return pc_no_plane, pca, sampling_factor
