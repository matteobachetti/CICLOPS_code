import numba
from numba import types
from numba.extending import overload_method
import numpy as np

from .base import jit, njit, vectorize, float32, float64, int32, int64
from .ffa import start_value, shift, roll

try:
    from tqdm import tqdm as show_progress
except ImportError:
    def show_progress(a):
        return a


@vectorize([(float64, float64),
            (int64, int64),
            (float32, float32),
            (int32, int32)])
def sum_arrays(arr1, arr2):
    return arr1 + arr2


def ffa_step(array, step, ntables):
    array_reshaped_dum = np.copy(array)
    jump = 2 ** step

    for prof_n in range(ntables):
        start = start_value(prof_n, step)
        sh = shift(prof_n, step)
        jumpstart = start + jump
        if sh > 0:
            rolled = np.roll(array[start + jump, :], -sh, axis=0)
            array_reshaped_dum[prof_n, :] = \
                sum_arrays(array[start, :], rolled[:])
        else:
            array_reshaped_dum[prof_n, :] = \
                sum_arrays(array[start, :], array[jumpstart, :])

    return array_reshaped_dum


def local_ffa(array, bin_period, z_n_n=2):
    """Fast folding algorithm search
    """
    N_raw = len(array)
    ntables = np.int(2**np.ceil(np.log2(N_raw // bin_period + 1)))
    if ntables <= 1: return np.zeros(1), np.zeros(1)
    N = ntables * bin_period
    new_arr = np.zeros((N, array.shape[1]))
    new_arr[:N_raw, :] = array

    array_reshaped = new_arr.reshape([ntables, bin_period, array.shape[1]])

    for step in range(0, np.int(np.log2(ntables))):
        array_reshaped = ffa_step(array_reshaped, step, ntables)

        for arr in array_reshaped:
            # This yields the profiles. Will need to substitute it with
            # some communication function
            print(arr)


def _quick_rebin(counts, current_rebin):
    """

    Examples
    --------
    >>> counts = np.arange(1, 11)
    >>> reb = _quick_rebin(counts, 2)
    >>> np.allclose(reb, [3, 7, 11, 15, 19])
    True
    >>> counts = np.arange(1, 11).reshape([10, 1])
    >>> reb = _quick_rebin(counts, 2)
    >>> np.allclose(reb, [[3], [7], [11], [15], [19]])
    True
    >>> counts = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    >>> reb = _quick_rebin(counts, 2)
    >>> np.allclose(reb, [[3, 3], [7, 7]])
    True
    """
    n = int(counts.shape[0] // current_rebin)
    newshape = (n, current_rebin)
    if len(counts.shape) == 2:
        newshape = (n, current_rebin, counts.shape[1])

    rebinned_counts = np.sum(
        counts[:n * current_rebin].reshape(newshape), axis=1)
    return rebinned_counts


def produce_profiles(samples, dt, period_min, period_max):
    pmin = np.floor(period_min / dt)
    pmax = np.ceil(period_max / dt)
    bin_periods = None
    stats = None

    current_rebin = 1
    rebinned_samples = samples
    for bin_period in show_progress(np.arange(pmin, pmax + 1, dtype=int)):
        # Only powers of two
        rebin = int(2**np.floor(np.log2(bin_period / pmin)))
        if rebin > current_rebin:
            current_rebin = rebin
            rebinned_samples = _quick_rebin(rebinned_samples, current_rebin)

        # When rebinning, bin_period // current_rebin is the same for nearby
        # periods
        if bin_period % current_rebin != 0:
            continue

        per, st = local_ffa(rebinned_samples, bin_period // current_rebin)

        per *= current_rebin

        if per[0] == 0: continue
        elif bin_periods is None:
            bin_periods = per[:-1] * dt
            stats = st[:-1]
        else:
            bin_periods = np.concatenate((bin_periods, per[:-1] * dt))
            stats = np.concatenate((stats, st[:-1]))

    return bin_periods, stats


if __name__ == '__main__':
    samples = np.random.poisson((10000, 16))
    produce_profiles(samples, 0.1, 3., 10)
