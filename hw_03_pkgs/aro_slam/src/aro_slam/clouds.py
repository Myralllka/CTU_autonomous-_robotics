from __future__ import absolute_import, division, print_function
from enum import Enum
import numpy as np
try:
    from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
except ImportError:
    from .compat import structured_to_unstructured, unstructured_to_structured


__all__ = [
    'Channels',
    'descriptor',
    'DType',
    'e2p',
    'normal',
    'p2e',
    'position',
    'transform',
    'viewpoint'
]


class Channels(Enum):
    position = ['x', 'y', 'z']
    mean = ['mean_%s' % f for f in 'xyz']
    normal = ['normal_%s' % f for f in 'xyz']
    curvature = ['curvature']
    roughness = ['roughness']
    viewpoint = ['vp_%s' % f for f in 'xyz']
    projection = ['proj_%s' % f for f in 'xyz']


position_dtype = [(f, 'f4') for f in Channels.position.value]
mean_dtype = [(f, 'f4') for f in Channels.mean.value]
normal_dtype = [(f, 'f4') for f in Channels.normal.value]
curvature_dtype = [(f, 'f4') for f in Channels.curvature.value]
roughness_dtype = [(f, 'f4') for f in Channels.roughness.value]
viewpoint_dtype = [(f, 'f4') for f in Channels.viewpoint.value]
projection_dtype = [(f, 'f4') for f in Channels.projection.value]


class DType(Enum):
    position = np.dtype(position_dtype)
    mean = np.dtype(mean_dtype)
    normal = np.dtype(normal_dtype)
    curvature = np.dtype(curvature_dtype)
    roughness = np.dtype(roughness_dtype)
    viewpoint = np.dtype(viewpoint_dtype)
    projection = np.dtype(projection_dtype)


def position(x_struct):
    return structured_to_unstructured(x_struct[Channels.position.value])


def normal(x_struct):
    return structured_to_unstructured(x_struct[Channels.normal.value])


def viewpoint(x_struct):
    return structured_to_unstructured(x_struct[Channels.viewpoint.value])


def e2p(x, axis=-1):
    assert isinstance(x, np.ndarray)
    assert isinstance(axis, int)
    h_size = list(x.shape)
    h_size[axis] = 1
    h = np.ones(h_size, dtype=x.dtype)
    xh = np.concatenate((x, h), axis=axis)
    return xh


def p2e(xh, axis=-1):
    assert isinstance(xh, np.ndarray)
    assert isinstance(axis, int)
    if axis != -1:
        xh = xh.swapaxes(axis, -1)
    x = xh[..., :-1]
    if axis != -1:
        x = x.swapaxes(axis, -1)
    return x


def transform(T, x_struct):
    assert isinstance(T, np.ndarray)
    assert T.shape == (4, 4)
    assert isinstance(x_struct, np.ndarray)
    x_struct = x_struct.copy()
    fields_op = []
    for fs, dtype, op in ((Channels.position.value, DType.position.value, 'Rt'),
                   (Channels.viewpoint.value, DType.viewpoint.value, 'Rt'),
                   (Channels.normal.value, DType.normal.value, 'R')):
        if fs[0] in x_struct.dtype.fields:
            fields_op.append((fs, op))
    for fs, op in fields_op:
        x = structured_to_unstructured(x_struct[fs])
        if op == 'Rt':
            x = p2e(np.matmul(e2p(x), T.T))
        elif op == 'R':
            x = np.matmul(x, T[:-1, :-1].T)
        x_str = unstructured_to_structured(x, dtype=dtype)
        for i in range(len(fs)):
            x_struct[fs[i]] = x_str[dtype.names[i]]
    return x_struct


def descriptor(x_struct, fields=None, weights=None):
    if fields is None:
        fields = Channels.position.value + Channels.normal.value
    assert weights is None or len(fields) == len(weights)
    x = structured_to_unstructured(x_struct[fields])
    if weights is not None:
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        weights = weights.reshape((1, -1))
        x = x * weights
    return x
