from __future__ import absolute_import, division, print_function
from enum import Enum
from .clouds import e2p, normal, p2e, position, transform
import numpy as np
try:
    from numpy.lib.recfunctions import unstructured_to_structured
except ImportError:
    from .compat import unstructured_to_structured
import rospy
from scipy.spatial import cKDTree
import unittest

__all__ = [
    'absolute_orientation',
    'icp',
    'IcpResult',
    'Loss',
]


class Loss(Enum):
    point_to_plane = 'point_to_plane'
    point_to_point = 'point_to_point'




def absolute_orientation(x, y):
    """Find transform R, t between x and y, such that the sum of squared
    distances ||R * x[:, i] + t - y[:, i]|| is minimum.

    :param x: Points to align, D-by-H array.
    :param y: Reference points to align to, D-by-H array.

    :return: Optimized transform from SE(D) as (D+1)-by-(D+1) array,
        T = [R t; 0... 1].
    """
    assert x.shape == y.shape, 'Inputs must be same size.'
    assert x.shape[1] > 0
    assert y.shape[1] > 0
    d = x.shape[0]
    T = np.eye(d + 1)
    # TODO: ARO homework 4: Implement absolute orientation.
    return T


class IcpResult(object):
    """ICP registration result."""
    def __init__(self, T=None, num_iters=None, idx=None, inliers=None,
                 x_inliers=None, y_inliers=None, mean_inlier_dist=None):
        """Initialize ICP registration result.

        @param T A 4-by-4 aligning transform array or None if registration failed.
        @param num_iters Number of iterations run.
        @param idx Nearest neighbors from descriptor matching.
        @param x_inliers Aligned M-by-3 position array of inliers.
        @param y_inliers Reference M-by-3 position array of inliers.
                In general, not original y for point-to-plane loss.
        @param inliers Inlier mask, use x[inliers] or y[idx[inliers]].
        @param mean_inlier_dist Mean point distance for inlier correspondences.
        """
        self.T = T
        self.num_iters = num_iters
        self.idx = idx
        self.x_inliers = x_inliers
        self.y_inliers = y_inliers
        self.inliers = inliers
        self.mean_inlier_dist = mean_inlier_dist


def icp(x_struct, y_struct, y_index=None,
        descriptor=position,
        T=None, max_iters=50,
        inlier_ratio=1.0, inlier_dist_mult=1.0, max_inlier_dist=float('inf'),
        loss=Loss.point_to_point):
    """Iterative closest point (ICP) algorithm, minimizing sum of squared
    point-to-point or point-to-plane distances.

    Input point clouds are structured arrays, with
    - position fields 'x', 'y', 'z', and
    - normal fields 'normal_x', 'normal_y', 'normal_z'.

    @param x_struct: Points to align, structured array.
    @param y_struct: Reference points to align to.
    @param y_index: NN search index for y_struct, cKDTree, which can be queried
            with descriptors(x_struct).
    @param descriptor: callable to create descriptors from structured array.
    @param max_iters: Maximum number of iterations.
    @param inlier_ratio: Ratio of inlier correspondences with lowest distances
            for which we optimize the criterion in given iteration. The inliers
            set may change each iteration.
    @param inlier_dist_mult: Multiplier of the maximum inlier distance found
            using inlier ratio above, enlarging or reducing the inlier set for
            optimization.
    @param max_inlier_dist: Maximum distance for inlier correspondence.
    @param T: Initial transform estimate from SE(D), defaults to identity.
    @return: IcpResult Optimized transform from SE(D) as (D+1)-by-(D+1) array,
            mean inlier distace from the last iteration, and
            boolean inlier mask from the last iteration, x[inl] are the
            inlier points.
    """
    assert isinstance(x_struct, np.ndarray) and x_struct.shape[0] > 0
    assert isinstance(y_struct, np.ndarray) and y_struct.shape[0] > 0
    assert y_index is None or isinstance(y_index, cKDTree)
    assert callable(descriptor)
    assert T is None or isinstance(T, np.ndarray)
    assert max_iters > 0
    assert 0.0 <= inlier_ratio <= 1.0
    assert 0.0 < inlier_dist_mult

    n = 3 if 'z' in x_struct.dtype.names else 2

    if y_index is None:
        y_desc = descriptor(y_struct)
        y_index = cKDTree(y_desc)

    if T is None:
        T = np.eye(n + 1)

    assert T.shape == (n + 1, n + 1)

    # Boolean inlier mask from current iteration.
    inl = np.zeros((x_struct.size,), dtype=np.bool)
    # Mean inlier distance history (to assess improvement).
    inl_errs = []

    idx = None
    x_inl = None
    y_inl = None

    for iter in range(max_iters):
        # TODO: ARO homework 4: Implement point-to-point ICP.
        # TODO: ARO homework 4: Implement point-to-plane ICP.

        # 1. Transform source points to align with reference points

        # 2. Find correspondences (Nearest Neighbors Search)
        # Find distances between source and reference point clouds and corresponding indexes
        # (Hint: use https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)
        dists, idx = np.full(len(x_struct), np.nan), np.full(len(y_struct), np.nan)


        # 3. Construct set of inliers (median filter, implemented :))
        d_max = np.percentile(dists, 100.0 * inlier_ratio)
        # Use fixed sample size for assessing improvement.
        inl = dists <= d_max
        inl_errs.append(dists[inl].mean())
        # Use adaptive sample size for optimization.
        d_max = inlier_dist_mult * d_max
        d_max = min(d_max, max_inlier_dist)
        inl = dists <= d_max
        inl_ratios.append(inl.mean())
        n_inliers = inl.sum()
        if n_inliers == 0:
            rospy.logwarn('Not enough inliers: %i.', n_inliers)
            break

        # 4. Use inlier points with found correspondences to solve Absolute orientation

        # 5. Stop the ICP loop when the inliers error does not change much

    else:
        rospy.logwarn('Max iter. %i: inliers: %.2f, mean dist.: %.3g, max dist: %.3g.',
                      max_iters, inl.mean(), inl_errs[-1], d_max)

    return IcpResult(T=T, num_iters=iter, idx=idx, inliers=inl,
                     x_inliers=x_inl, y_inliers=y_inl,
                     mean_inlier_dist=inl_errs[-1] if inl_errs else float('nan'))



def icp_demo():
    from numpy.lib.recfunctions import structured_to_unstructured
    import os
    import matplotlib.pyplot as plt

    def read_poses(path):
        poses = np.genfromtxt(path, delimiter=', ', skip_header=True)
        ids = np.genfromtxt(path, delimiter=', ', dtype=str, skip_header=True)[:, 0].tolist()
        # assert ids == list(range(len(ids)))
        poses = poses[:, 2:]
        poses = poses.reshape((-1, 4, 4))
        poses = dict(zip(ids, poses))
        return poses

    def read_cloud(npz_file):
        cloud = np.load(npz_file)['cloud']
        if cloud.ndim == 2:
            cloud = cloud.reshape((-1,))
        return cloud

    def filter_grid(cloud, grid_res, preserve_order=False, log=False, rng=np.random.default_rng(135)):
        """Keep single point within each cell. Order is not preserved."""
        assert isinstance(cloud, np.ndarray)
        assert isinstance(grid_res, float) and grid_res > 0.0

        # Convert to numpy array with positions.
        if cloud.dtype.names:
            x = structured_to_unstructured(cloud[['x', 'y', 'z']])
        else:
            x = cloud

        # Create voxel indices.
        keys = np.floor(x / grid_res).astype(int).tolist()

        # Last key will be kept, shuffle if needed.
        # Create index array for tracking the input points.
        ind = list(range(len(keys)))

        # Make the last item random.
        rng.shuffle(ind)
        # keys = keys[ind]
        keys = [keys[i] for i in ind]

        # Convert to immutable keys (tuples).
        keys = [tuple(i) for i in keys]

        # Dict keeps the last value for each key (already reshuffled).
        key_to_ind = dict(zip(keys, ind))
        if preserve_order:
            ind = sorted(key_to_ind.values())
        else:
            ind = list(key_to_ind.values())

        if log:
            # print('%.3f = %i / %i points kept (grid res. %.3f m).'
            #       % (mask.double().mean(), mask.sum(), mask.numel(), grid_res))
            print('%.3f = %i / %i points kept (grid res. %.3f m).'
                  % (len(ind) / len(keys), len(ind), len(keys), grid_res))

        filtered = cloud[ind]
        return filtered

    def visualize_clouds(x_struct, y_struct, **kwargs):
        plt.figure(figsize=(10, 10))
        plt.plot(x_struct['x'], x_struct['y'], '.', markersize=0.3, label='source cloud')
        plt.plot(y_struct['x'], y_struct['y'], '.', markersize=0.3, label='target cloud')
        plt.grid()
        plt.axis('equal')
        plt.legend()
        plt.show()

    # load the data: 2 point clouds and their ground truth poses from a data set
    # wget http://ptak.felk.cvut.cz/vras/data/fee_corridor/sequences/seq2/poses/poses.csv
    # wget http://ptak.felk.cvut.cz/vras/data/fee_corridor/sequences/seq2/ouster_points/1669300804_715071232.npz
    # wget http://ptak.felk.cvut.cz/vras/data/fee_corridor/sequences/seq2/ouster_points/1669300806_15306496.npz

    # load cloud poses
    path = os.path.dirname(__file__)
    poses = read_poses(os.path.join(path, 'poses.csv'))
    id1, id2 = '1669300804_715071232', '1669300806_15306496'
    pose1 = poses[id1]
    pose2 = poses[id2]

    # load point clouds
    cloud1 = read_cloud(os.path.join(path, '%s.npz' % id1))
    cloud2 = read_cloud(os.path.join(path, '%s.npz' % id2))

    # apply grid filtering to point clouds
    cloud1 = filter_grid(cloud1, grid_res=0.1)
    cloud2 = filter_grid(cloud2, grid_res=0.1)

    # visualize not aligned point clouds
    visualize_clouds(cloud1, cloud2)

    # ground truth transformation that aligns the point clouds (from data set)
    Tr_gt = np.matmul(np.linalg.inv(pose2), pose1)

    # run ICP algorithm to estimate the transformation (it is initialized with identity matrix)
    Tr_init = np.eye(4)
    res = icp(cloud1, cloud2, T=Tr_init, inlier_ratio=0.9, inlier_dist_mult=2.0, max_iters=100,
              loss=Loss.point_to_point, descriptor=position)
    Tr_icp = res.T
    # assert np.allclose(Tr_icp, Tr_gt, atol=1e-2)

    print('ICP found transformation:\n%s\n' % Tr_icp)
    print('GT transformation:\n%s\n' % Tr_gt)
    print('ICP mean inliers distance: %.3f [m]' % res.mean_inlier_dist)

    # visualize the clouds after ICP alignment with found transformation
    visualize_clouds(transform(Tr_icp, cloud1), cloud2)


if __name__ == '__main__':
    # python -m aro_slam.icp
    # main()
    icp_demo()
