import math
import numpy as np

# The following are imports from library tf.transformations included here so that this library can be run without ROS

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def euler_from_matrix(matrix, axes='sxyz'):
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


# https://stackoverflow.com/a/24837438/1076564
def merge_dicts(dict1, dict2):
    """Recursively merges dict2 into dict1."""
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2
    for k in dict2:
        if k in dict1:
            dict1[k] = merge_dicts(dict1[k], dict2[k])
        else:
            dict1[k] = dict2[k]
    return dict1


def coords_to_transform_matrix(x_y_yaw):
    """Converts the x, y and yaw 2D coordinates to 3D homogeneous transformation matrix.

    :param x_y_yaw: A 3-tuple x, y, yaw either as a Python iterable or as Numpy array 3x1 or 3.
    :type x_y_yaw: np.ndarray or list or tuple.
    :return: The transformation matrix, numpy array 4x4.
    :rtype: np.ndarray
    """
    x, y, yaw = x_y_yaw.ravel().tolist() if isinstance(x_y_yaw, np.ndarray) else x_y_yaw
    c = np.cos(yaw)
    s = np.sin(yaw)
    ma_in_body = np.array([
        [c, -s, 0, x],
        [s,  c, 0, y],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ])
    return ma_in_body


def transform_matrix_to_coords(matrix):
    """Convert 4x4 homogeneous 3D transformation matrix to 2D coordinates x, y, yaw.

    :param np.ndarray matrix: The 4x4 transformation matrix.
    :return: x, y, yaw
    :rtype: tuple
    """
    return matrix[0, 3], matrix[1, 3], euler_from_matrix(matrix)[2]


def find_nearest_index_for_new_measurement(stamps, stamp=None):
    """Find the best index into stamps for a measurement with the given timestamp.

    :param list stamps: The list of all available timestamps. List of int nanoseconds.
    :param stamp: The timestamp in nanoseconds. If `None`, the measurement is inserted to the latest time.
    :type stamp: int or None
    :return: The index. `None` is returned if there are no odom measurements yet and thus there is no place to put
             the measurement. Number 0 to N.
    :rtype: int or None
    """
    # no measurements yet -> we don't have anywhere to put the message
    if len(stamps) == 0:
        return None

    # no timestamp -> always add to the last position
    # only one odom measurement -> add to the only position
    if stamp is None or len(stamps) <= 1:
        return len(stamps) - 1

    # timestamp is newer than the newest measurement -> add to the last position
    if stamp >= stamps[-1]:
        return len(stamps) - 1

    # from now on, we have at least two measurements, and timestamp is not newer than the newest measurement

    # we perform a linear search from the end of the list to find the best slot
    # this is generally not very good for performance, but we assume we almost always find a slot near the end,
    # so it would not be worth it to implement binary search here
    i = len(stamps) - 1
    while stamp < stamps[i] and i > 0:
        i -= 1

    # i now points to the first slot where timestamp is older than the measurement
    # so i + 1 is the first slot where timestamp is newer than the measurement
    diff_older = abs(stamp - stamps[i])
    diff_newer = abs(stamp - stamps[i + 1])

    # choose the closer slot
    if diff_older < diff_newer:
        return i
    return i + 1


def sec_to_nsec(sec):
    """Return the number in seconds as nanoseconds.

    :param sec: Seconds. Passing large float values will lead to precision loss.
    :type sec: float or int
    :return: Nanoseconds.
    :rtype: int
    """
    return int(sec * int(1e9))


def nsec_to_sec(nsec):
    """Return the number in nanoseconds as fractional seconds.

    :param int nsec: Nanoseconds. Passing too large values (larger than a few days) will lead to precision loss.
    :return: Seconds.
    :rtype: float
    """
    return nsec / 1e9
