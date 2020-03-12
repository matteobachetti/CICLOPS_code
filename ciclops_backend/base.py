# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""A miscellaneous collection of basic functions."""

import sys
import copy
import os
import warnings
from functools import wraps
from collections.abc import Iterable
from pathlib import Path
import tempfile

import numpy as np
from numba import jit, njit, prange, vectorize
from numba import float32, float64, int32, int64


def _assign_value_if_none(value, default):
    if value is None:
        return default
    return value


def _look_for_array_in_array(array1, array2):
    """
    Examples
    --------
    >>> _look_for_array_in_array([1, 2], [2, 3, 4])
    2
    >>> _look_for_array_in_array([1, 2], [3, 4, 5]) is None
    True
    """
    for a1 in array1:
        if a1 in array2:
            return a1
    return None


def is_string(s):
    """Portable function to answer this question."""
    return isinstance(s, str)  # NOQA


def _order_list_of_arrays(data, order):
    """
    Examples
    --------
    >>> order = [1, 2, 0]
    >>> new = _order_list_of_arrays({'a': [4, 5, 6], 'b':[7, 8, 9]}, order)
    >>> np.all(new['a'] == [5, 6, 4])
    True
    >>> np.all(new['b'] == [8, 9, 7])
    True
    >>> new = _order_list_of_arrays([[4, 5, 6], [7, 8, 9]], order)
    >>> np.all(new[0] == [5, 6, 4])
    True
    >>> np.all(new[1] == [8, 9, 7])
    True
    >>> _order_list_of_arrays(2, order) is None
    True
    """
    if hasattr(data, 'items'):
        data = dict((i[0], np.asarray(i[1])[order])
                    for i in data.items())
    elif hasattr(data, 'index'):
        data = [np.asarray(i)[order] for i in data]
    else:
        data = None
    return data


def mkdir_p(path):
    """Safe mkdir function."""
    return os.makedirs(path, exist_ok=True)


@njit(nogil=True, parallel=False)
def _get_bin_edges(a, bins, a_min, a_max):
    bin_edges = np.zeros(bins+1, dtype=np.float64)

    delta = (a_max - a_min) / bins
    for i in range(bin_edges.size):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


def get_bin_edges(a, bins):
    """

    Examples
    --------
    >>> array = np.array([0., 10.])
    >>> bins = 2
    >>> np.allclose(get_bin_edges(array, bins), [0, 5, 10])
    True
    """
    a_min = np.min(a)
    a_max = np.max(a)
    return _get_bin_edges(a, bins, a_min, a_max)


@njit(nogil=True, parallel=False)
def compute_bin(x, bin_edges):
    """

    Examples
    --------
    >>> bin_edges = np.array([0, 5, 10])
    >>> compute_bin(1, bin_edges)
    0
    >>> compute_bin(5, bin_edges)
    1
    >>> compute_bin(10, bin_edges)
    1
    """

    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@njit(nogil=True, parallel=False)
def _hist1d_numba_seq(H, tracks, bins, ranges):
    delta = 1 / ((ranges[1] - ranges[0]) / bins)

    for t in range(tracks.size):
        i = (tracks[t] - ranges[0]) * delta
        if 0 <= i < bins:
            H[int(i)] += 1

    return H


def hist1d_numba_seq(a, bins, ranges, use_memmap=False, tmp=None):
    """
    Examples
    --------
    >>> if os.path.exists('out.npy'): os.unlink('out.npy')
    >>> x = np.random.uniform(0., 1., 100)
    >>> H, xedges = np.histogram(x, bins=5, range=[0., 1.])
    >>> Hn = hist1d_numba_seq(x, bins=5, ranges=[0., 1.], tmp='out.npy',
    ...                       use_memmap=True)
    >>> assert np.all(H == Hn)
    >>> # The number of bins is small, memory map was not used!
    >>> assert not os.path.exists('out.npy')
    >>> H, xedges = np.histogram(x, bins=10**8, range=[0., 1.])
    >>> Hn = hist1d_numba_seq(x, bins=10**8, ranges=[0., 1.], tmp='out.npy',
    ...                       use_memmap=True)
    >>> assert np.all(H == Hn)
    >>> assert os.path.exists('out.npy')
    """
    if bins > 10**7 and use_memmap:
        if tmp is None:
            tmp = tempfile.NamedTemporaryFile('w+')
        hist_arr = np.lib.format.open_memmap(
            tmp, mode='w+', dtype=a.dtype, shape=(bins,))
    else:
        hist_arr = np.zeros((bins,), dtype=a.dtype)

    return _hist1d_numba_seq(hist_arr, a, bins, np.asarray(ranges))


@njit(nogil=True, parallel=False)
def _hist2d_numba_seq(H, tracks, bins, ranges):
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        i = (tracks[0, t] - ranges[0, 0]) * delta[0]
        j = (tracks[1, t] - ranges[1, 0]) * delta[1]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j)] += 1

    return H


def hist2d_numba_seq(x, y, bins, ranges):
    """
    Examples
    --------
    >>> x = np.random.uniform(0., 1., 100)
    >>> y = np.random.uniform(2., 3., 100)
    >>> H, xedges, yedges = np.histogram2d(x, y, bins=(5, 5),
    ...                                    range=[(0., 1.), (2., 3.)])
    >>> Hn = hist2d_numba_seq(x, y, bins=(5, 5),
    ...                       ranges=[[0., 1.], [2., 3.]])
    >>> assert np.all(H == Hn)
    """
    H = np.zeros((bins[0], bins[1]), dtype=np.uint64)
    return _hist2d_numba_seq(H, np.array([x, y]), np.asarray(list(bins)),
                             np.asarray(ranges))


@njit(nogil=True, parallel=False)
def _hist3d_numba_seq(H, tracks, bins, ranges):
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        i = (tracks[0, t] - ranges[0, 0]) * delta[0]
        j = (tracks[1, t] - ranges[1, 0]) * delta[1]
        k = (tracks[2, t] - ranges[2, 0]) * delta[2]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j), int(k)] += 1

    return H


def hist3d_numba_seq(tracks, bins, ranges):
    """
    Examples
    --------
    >>> x = np.random.uniform(0., 1., 100)
    >>> y = np.random.uniform(2., 3., 100)
    >>> z = np.random.uniform(4., 5., 100)
    >>> H, _ = np.histogramdd((x, y, z), bins=(5, 6, 7),
    ...                       range=[(0., 1.), (2., 3.), (4., 5)])
    >>> Hn = hist3d_numba_seq((x, y, z), bins=(5, 6, 7),
    ...                       ranges=[[0., 1.], [2., 3.], [4., 5.]])
    >>> assert np.all(H == Hn)
    """

    H = np.zeros((bins[0], bins[1], bins[2]), dtype=np.uint64)
    return _hist3d_numba_seq(H, np.asarray(tracks), np.asarray(list(bins)),
                             np.asarray(ranges))


@njit(nogil=True, parallel=False)
def _hist2d_numba_seq_weight(H, tracks, weights, bins, ranges):
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        i = (tracks[0, t] - ranges[0, 0]) * delta[0]
        j = (tracks[1, t] - ranges[1, 0]) * delta[1]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j)] += weights[t]

    return H


def hist2d_numba_seq_weight(x, y, weights, bins, ranges):
    """
    Examples
    --------
    >>> x = np.random.uniform(0., 1., 100)
    >>> y = np.random.uniform(2., 3., 100)
    >>> weight = np.random.uniform(0, 1, 100)
    >>> H, xedges, yedges = np.histogram2d(x, y, bins=(5, 5),
    ...                                    range=[(0., 1.), (2., 3.)],
    ...                                    weights=weight)
    >>> Hn = hist2d_numba_seq_weight(x, y, bins=(5, 5),
    ...                              ranges=[[0., 1.], [2., 3.]],
    ...                              weights=weight)
    >>> assert np.all(H == Hn)
    """
    H = np.zeros((bins[0], bins[1]), dtype=np.double)
    return _hist2d_numba_seq_weight(
        H, np.array([x, y]), weights, np.asarray(list(bins)),
        np.asarray(ranges))


@njit(nogil=True, parallel=False)
def _hist3d_numba_seq_weight(H, tracks, weights, bins, ranges):
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        i = (tracks[0, t] - ranges[0, 0]) * delta[0]
        j = (tracks[1, t] - ranges[1, 0]) * delta[1]
        k = (tracks[2, t] - ranges[2, 0]) * delta[2]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j), int(k)] += weights[t]

    return H


def hist3d_numba_seq_weight(tracks, weights, bins, ranges):
    """
    Examples
    --------
    >>> x = np.random.uniform(0., 1., 100)
    >>> y = np.random.uniform(2., 3., 100)
    >>> z = np.random.uniform(4., 5., 100)
    >>> weights = np.random.uniform(0, 1., 100)
    >>> H, _ = np.histogramdd((x, y, z), bins=(5, 6, 7),
    ...                       range=[(0., 1.), (2., 3.), (4., 5)],
    ...                       weights=weights)
    >>> Hn = hist3d_numba_seq_weight(
    ...    (x, y, z), weights, bins=(5, 6, 7),
    ...    ranges=[[0., 1.], [2., 3.], [4., 5.]])
    >>> assert np.all(H == Hn)
    """

    H = np.zeros((bins[0], bins[1], bins[2]), dtype=np.double)
    return _hist3d_numba_seq_weight(
        H, np.asarray(tracks), weights, np.asarray(list(bins)),
        np.asarray(ranges))


@njit(nogil=True, parallel=False)
def index_arr(a, ix_arr):
    strides = np.array(a.strides) / a.itemsize
    ix = int((ix_arr * strides).sum())
    return a.ravel()[ix]


@njit(nogil=True, parallel=False)
def index_set_arr(a, ix_arr, val):
    strides = np.array(a.strides) / a.itemsize
    ix = int((ix_arr * strides).sum())
    a.ravel()[ix] = val


@njit(nogil=True, parallel=False)
def _histnd_numba_seq(H, tracks, bins, ranges, slice_int):
    delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        slicearr = np.array([(tracks[dim, t] - ranges[dim, 0]) * delta[dim]
                             for dim in range(tracks.shape[0])])

        good = np.all((slicearr < bins) & (slicearr >= 0))
        slice_int[:] = slicearr

        if good:
            curr = index_arr(H, slice_int)
            index_set_arr(H, slice_int, curr + 1)

    return H


def histnd_numba_seq(tracks, bins, ranges):
    """
    Examples
    --------
    >>> x = np.random.uniform(0., 1., 100)
    >>> y = np.random.uniform(2., 3., 100)
    >>> z = np.random.uniform(4., 5., 100)
    >>> # 2d example
    >>> H, _, _ = np.histogram2d(x, y, bins=np.array((5, 5)),
    ...                          range=[(0., 1.), (2., 3.)])
    >>> alldata = np.array([x, y])
    >>> Hn = histnd_numba_seq(alldata, bins=np.array([5, 5]),
    ...                       ranges=np.array([[0., 1.], [2., 3.]]))
    >>> assert np.all(H == Hn)
    >>> # 3d example
    >>> H, _ = np.histogramdd((x, y, z), bins=np.array((5, 6, 7)),
    ...                       range=[(0., 1.), (2., 3.), (4., 5)])
    >>> alldata = np.array([x, y, z])
    >>> Hn = hist3d_numba_seq(alldata, bins=np.array((5, 6, 7)),
    ...                       ranges=np.array([[0., 1.], [2., 3.], [4., 5.]]))
    >>> assert np.all(H == Hn)
    """
    H = np.zeros(tuple(bins), dtype=np.uint64)
    slice_int = np.zeros(len(bins), dtype=np.uint64)

    return _histnd_numba_seq(H, tracks, bins, ranges, slice_int)
