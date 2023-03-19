from __future__ import absolute_import, division, print_function
import numpy as np
import rospy
from timeit import default_timer as timer


__all__ = [
    'array',
    'inverse_affine',
    'logistic',
    'logit',
    'rotation_angle',
    'slots',
    'timer',
    'timing'
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
