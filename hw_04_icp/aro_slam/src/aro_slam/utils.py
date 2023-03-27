from __future__ import absolute_import, division, print_function
import numpy as np
import rospy
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    from numpy.lib.recfunctions import unstructured_to_structured, structured_to_unstructured
except ImportError:
    from .compat import unstructured_to_structured, structured_to_unstructured


__all__ = [
    'array',
    'inverse_affine',
    'logistic',
    'logit',
    'rotation_angle',
    'slots',
    'timer',
    'timing',
    'visualize_clouds_3d',
    'filter_grid',
    'visualize_clouds_2d'
]


def timing(f):
    def timing_wrapper(*args, **kwargs):
        t0 = timer()
        try:
            ret = f(*args, **kwargs)
            return ret
        finally:
            t1 = timer()
            rospy.loginfo('%s %.6f s' % (f.__name__, t1 - t0))
    return timing_wrapper


def slots(msg):
    """Return message attributes (slots) as list."""
    return [getattr(msg, var) for var in msg.__slots__]


def array(msg):
    """Return message attributes (slots) as array."""
    return np.array(slots(msg))


def col(arr):
    """Convert array to column vector."""
    assert isinstance(arr, np.ndarray)
    return arr.reshape((arr.size, 1))


def logistic(x):
    """Standard logistic function, inverse of logit function."""
    return 1. / (1. + np.exp(-x))


def logit(p):
    """Logit function or the log-odds, inverse of logistic function.
    The logarithm of the odds p / (1 - p) where p is probability."""
    return np.log(p / (1. - p))


def inverse_affine(T):
    """Invert affine transform, [R t]^-1 = [R^T -R^T*t]."""
    assert isinstance(T, np.ndarray)
    assert T.ndim >= 2
    d = T.shape[-1] - 1
    assert d <= T.shape[-2] <= d + 1
    R = T[:d, :d]
    t = T[:d, d:]
    T_inv = T.copy()
    # T_inv = np.eye(d + 1)
    T_inv[..., :d, :d] = R.T
    T_inv[..., :d, d:] = -np.matmul(R.T, t)
    return T_inv


def rotation_angle(R):
    """Rotation angle from a rotation matrix."""
    assert isinstance(R, np.ndarray)
    assert R.ndim >= 2
    assert R.shape[-2] == R.shape[-1]
    d = R.shape[-1]
    alpha = np.arccos((np.trace(R, axis1=-2, axis2=-1) - (d - 2)) / 2.)
    return alpha


def visualize_clouds_2d(P, Q, **kwargs):
    if P.dtype.names:
        P = structured_to_unstructured(P[['x', 'y', 'z']])
    if Q.dtype.names:
        Q = structured_to_unstructured(Q[['x', 'y', 'z']])

    plt.figure()

    plt.plot(P[:, 0], P[:, 1], 'o', label='source cloud', **kwargs)
    plt.plot(Q[:, 0], Q[:, 1], 'x', label='target cloud', **kwargs)

    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.show()

def filter_grid(cloud, grid_res, log=False, rng=np.random.default_rng(135)):
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
    ind = list(key_to_ind.values())

    if log:
        print('%.3f = %i / %i points kept (grid res. %.3f m).'
              % (len(ind) / len(keys), len(ind), len(keys), grid_res))

    filtered = cloud[ind]
    return filtered

def visualize_clouds_3d(P, Q, **kwargs):
    if P.dtype.names:
        P = structured_to_unstructured(P[['x', 'y', 'z']])
    if Q.dtype.names:
        Q = structured_to_unstructured(Q[['x', 'y', 'z']])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(P[:, 0], P[:, 1], P[:, 2], 'o', label='source cloud', **kwargs)
    ax.plot(Q[:, 0], Q[:, 1], Q[:, 2], 'x', label='target cloud', **kwargs)

    set_axes_equal(ax)
    ax.grid()
    ax.legend()
    plt.show()


# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
