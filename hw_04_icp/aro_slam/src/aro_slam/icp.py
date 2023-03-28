from __future__ import absolute_import, division, print_function
from enum import Enum
from .clouds import e2p, normal, p2e, position, transform
import numpy as np
try:
    from numpy.lib.recfunctions import unstructured_to_structured, structured_to_unstructured
except ImportError:
    from .compat import unstructured_to_structured, structured_to_unstructured
import rospy
from scipy.spatial import cKDTree
import unittest
import copy

__all__ = [
    'absolute_orientation',
    'icp',
    'IcpResult',
    'Loss',
    'AbsorientDomain'
]


class Loss(Enum):
    point_to_plane = 'point_to_plane'
    point_to_point = 'point_to_point'


class AbsorientDomain(Enum):
    SE2 = 'SE2'
    SE3 = 'SE3'

def absolute_orientation(x, y, domain=AbsorientDomain.SE2):
    """Find transform R, t between x and y, such that the sum of squared
    distances ||R * x[:, i] + t - y[:, i]|| is minimum.

    :param x: Points to align, D-by-H array.
    :param y: Reference points to align to, D-by-H array.
    :param domain: SE2 or SE3.

    :return: Optimized transform from SE(D) as (4)-by-(4) array,
        T = [R t; 0... 1].
    """
    assert x.shape == y.shape, 'Inputs must be same size.'
    assert x.shape[1] > 0
    assert y.shape[1] > 0

    # ARO homework 4: Implement absolute orientation.

    def center_data(inp):
        center = np.expand_dims(inp.mean(axis=1), 1)
        return center, inp - center

    T = np.eye(4)
    if domain == AbsorientDomain.SE2:
        x = x[:2]
        y = y[:2]
        x_mean, x_centered = center_data(x.copy())
        y_mean, y_centered = center_data(y.copy())
        Hx2 = np.dot(x_centered[0], y_centered[0])
        Hy2 = np.dot(x_centered[1], y_centered[1])
        Hxy = np.dot(x_centered[0], y_centered[1])
        Hyx = np.dot(x_centered[1], y_centered[0])
        th = np.arctan((Hxy - Hyx) / (Hx2 + Hy2))
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        t = (y_mean - R @ x_mean)
        T[:2, :2] = R
        T[:2, 3:4] = t

    elif domain == AbsorientDomain.SE3:
        x = x[:3]
        y = y[:3]
        x_mean, x_centered = center_data(x.copy())
        y_mean, y_centered = center_data(y.copy())

        H = x_centered @ y_centered.T
        U, S, Vh = np.linalg.svd(H, full_matrices=False)
        R = Vh.T @ U.T
        t = y_mean - R @ x_mean
        T[:3, :3] = R
        T[:3, 3:4] = t

    else:
        print("ERROR!!!!!")
        return None

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
        loss=Loss.point_to_point,
        absorient_domain=AbsorientDomain.SE2):
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
    @param absorient_domain: Absolute orientation domain, SE(2) or SE(3).
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
    inl = np.zeros((x_struct.size,), dtype=bool)
    # Mean inlier distance history (to assess improvement).
    inl_errs = [np.inf]

    # Mean inliers ratios (relative to number of points in a source cloud) histoty.
    inl_ratios = [0]

    ids = None
    x_inl = None
    y_inl = None
    
    x_struct = x_struct.copy()
    Q = structured_to_unstructured(y_struct[['x', 'y', 'z']]).T
    P_init = structured_to_unstructured(x_struct[['x', 'y', 'z']]).T

    Q_normals = structured_to_unstructured(y_struct[['normal_x', 'normal_y', 'normal_z']]).T
    P_normals = structured_to_unstructured(x_struct[['normal_x', 'normal_y', 'normal_z']]).T
    
    R, t = np.eye(3), np.array([0, 0, 0]).reshape(3, 1)

    for i in range(max_iters):
        # ARO homework 4: Implement point-to-point ICP.
        # ARO homework 4: Implement point-to-plane ICP.

        # 1. Transform source points to align with reference points
        P_corrected = R @ P_init + t
        x_struct["x"] = P_corrected[0]
        x_struct["y"] = P_corrected[1]
        x_struct["z"] = P_corrected[2]
        if 'normal_x' in x_struct.dtype.names:
            P_normals_corrected = R @ P_normals
            x_struct['normal_x'] = P_normals_corrected[0]
            x_struct['normal_y'] = P_normals_corrected[1]
            x_struct['normal_z'] = P_normals_corrected[2]

        # 2. Find correspondences (Nearest Neighbors Search)
        # Find distances between source and reference point clouds and corresponding indexes
        # (Hint: use https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)

        dists, ids = y_index.query(descriptor(x_struct))
        
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
        correspondences = np.asarray([(i, j) for i, j in zip(range(P_corrected.shape[1]), ids)])[inl]

        P_c = P_init[:, correspondences[:, 0]]
        Q_c = Q[:, correspondences[:, 1]]

        if loss == Loss.point_to_point:
            T = absolute_orientation(P_c, Q_c, absorient_domain)

        elif loss == Loss.point_to_plane:

            dd = np.sum(Q_normals[:, correspondences[:, 1]] * (P_corrected[:, correspondences[:, 0]] - Q_c), axis=0)
            Q_plane = P_corrected[:, correspondences[:, 0]] - dd * Q_normals[:, correspondences[:, 1]]

            T = absolute_orientation(P_c, Q_plane, absorient_domain)
        R = T[:3, :3]
        t = T[:3, 3:4]

        # 5. Stop the ICP loop when the inliers error does not change much

        eps = 1e-5
        if np.abs(inl_errs[-2] - inl_errs[-1]) < eps:
            break
    else:
        rospy.logwarn('Max iter. %i: inliers: %.2f, mean dist.: %.3g, max dist: %.3g.',
                      max_iters, inl.mean(), inl_errs[-1], d_max)

    #import matplotlib.pyplot as plt
    #plt.plot(inl_errs)
    #plt.show()

    return IcpResult(T=T, num_iters=iter, idx=ids, inliers=inl,
                     x_inliers=x_inl, y_inliers=y_inl,
                     mean_inlier_dist=inl_errs[-1] if inl_errs else float('nan'))



def absorient_demo(known_corresps=True):
    from .utils import visualize_clouds_2d
    from scipy.spatial.transform import Rotation
    """
    A function to test implementation of Absolute Orientation algorithm.
    """
    # generate point clouds
    # initialize pertrubation rotation
    theta = np.pi / 4
    R_true = np.eye(3)
    R_true[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
    t_true = np.array([[-2], [5], [0]])
    Tr_gt = np.eye(4)
    Tr_gt[:3, :3] = R_true
    Tr_gt[:3, 3:] = t_true

    # Generate data as a list of 2d points
    num_points = 30
    true_data = np.zeros((3, num_points))
    true_data[0, :] = range(0, num_points)
    true_data[1, :] = 0.2 * true_data[0, :] * np.sin(0.5 * true_data[0, :])

    # Move the data
    moved_data = R_true.dot(true_data) + t_true

    # Add noise
    n = 0.5 * (np.random.random((3, num_points)) - 0.5)  # noise
    moved_data = moved_data + n

    # Assign to variables we use in formulas.
    Q = true_data
    P = moved_data

    # visualize not aligned point clouds
    visualize_clouds_2d(P.T, Q.T, markersize=4)

    # choose inliers for alignment
    n_inl = num_points // 10
    # n_inl = 2
    inl_maks = np.random.choice(range(num_points), n_inl)
    P_inl = P[:, inl_maks]
    if known_corresps:
        Q_inl = Q[:, inl_maks]
    else:
        Q_inl = Q[:, np.random.choice(range(num_points), n_inl)]

    # run absolute orientation algorithm to estimate the transformation
    Tr = absolute_orientation(P_inl, Q_inl, domain=AbsorientDomain.SE3)

    print('ICP found transformation:\n%s\n' % Tr)
    print('GT transformation:\n%s\n' % Tr_gt)

    # visualize the clouds after ICP alignment with found transformation
    P_aligned = np.matmul(Tr[:3, :3], P) + Tr[:3, 3:]
    visualize_clouds_2d(P_aligned.T, Q.T, markersize=4)


def icp_demo():
    import os
    import rospkg
    from .utils import visualize_clouds_3d, filter_grid
    from .io import read_cloud, read_poses
    """
    The function utilizes a pair of point cloud scans captured in an indoor corridor-like environment.
    We run ICP algorithm to find a transformation that aligns the pair of clouds.
    The result is being tested along the ground-truth transformation available in a data set.
    The sample data is provided with the `aro_slam` package. However, if you would like to test your implementation
    with another point clouds, follow the next instructions.
    
    In order to load the data (2 point clouds and their ground truth poses from a data set), run:
    ```bash
        wget http://ptak.felk.cvut.cz/vras/data/fee_corridor/sequences/seq2/poses/poses.csv
        wget http://ptak.felk.cvut.cz/vras/data/fee_corridor/sequences/seq2/ouster_points/1669300804_715071232.npz
        wget http://ptak.felk.cvut.cz/vras/data/fee_corridor/sequences/seq2/ouster_points/1669300806_15306496.npz
    ```
    
    The `poses.csv` file contains 6DoF poses for all the point clouds from the data set as 4x4 matrices (T = [R|t])
    and corresponding indices of the scans which encodes recording time stamps in the format {sec}_{nsec}.
    For more information about the data, please, refer to https://paperswithcode.com/dataset/fee-corridor.
    """
    # download necessary data: 2 point clouds and poses
    # path = os.path.join(rospkg.RosPack().get_path('aro_slam'), 'data', 'fee_corridor')
    path = os.path.normpath(os.path.join(__file__, '../../../', 'data', 'fee_corridor'))
    id1, id2 = '1669300991_618086656', '1669301026_319255296'

    if not os.path.exists(path):
        os.mkdir(path)
        url = 'http://ptak.felk.cvut.cz/vras/data/fee_corridor/sequences/seq2'
        os.system('wget %s/poses/poses.csv -P %s' % (url, path))
        os.system('wget %s/ouster_points/%s.npz -P %s' % (url, id1, path))
        os.system('wget %s/ouster_points/%s.npz -P %s' % (url, id2, path))

    # load cloud poses
    poses = read_poses(os.path.join(path, 'poses.csv'))
    pose1 = poses[id1]
    pose2 = poses[id2]

    # load point clouds
    cloud1 = read_cloud(os.path.join(path, '%s.npz' % id1))
    cloud2 = read_cloud(os.path.join(path, '%s.npz' % id2))

    # apply grid filtering to point clouds
    cloud1 = filter_grid(cloud1, grid_res=0.1)
    cloud2 = filter_grid(cloud2, grid_res=0.1)

    # visualize not aligned point clouds
    #visualize_clouds_3d(cloud1, cloud2, markersize=0.3)

    # ground truth transformation that aligns the point clouds (from data set)
    Tr_gt = np.matmul(np.linalg.inv(pose2), pose1)

    # run ICP algorithm to estimate the transformation (it is initialized with identity matrix)
    Tr_init = np.eye(4)
    res = icp(cloud1, cloud2, T=Tr_init, inlier_ratio=0.9, inlier_dist_mult=2.0, max_iters=100,
              loss=Loss.point_to_plane, descriptor=position, absorient_domain=AbsorientDomain.SE3)
              #loss=Loss.point_to_point, descriptor=position, absorient_domain=AbsorientDomain.SE3)
    Tr_icp = res.T

    print('ICP found transformation:\n%s\n' % Tr_icp)
    print('GT transformation:\n%s\n' % Tr_gt)
    print('ICP mean inliers distance: %.3f [m]' % res.mean_inlier_dist)

    # visualize the clouds after ICP alignment with found transformation
    visualize_clouds_3d(transform(Tr_icp, cloud1), cloud2, markersize=0.3)


if __name__ == '__main__':
    # python -m aro_slam.icp
    absorient_demo()
    #icp_demo()
