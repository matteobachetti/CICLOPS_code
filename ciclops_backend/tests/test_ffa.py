import numpy as np
from hendrics.efsearch import fit

from ciclops_backend.ffa import ffa_search

import pytest

# @pytest.mark.skipif('not HAS_NUMBA')
def test_ffa():
    period = 0.01
    pmin = 0.0095
    pmax = 0.0105
    dt = 10**int(np.log10(period)) / 256
    length = 1
    times = np.arange(0, length, dt)

    flux = 10 + np.cos(2 * np.pi * times / period)

    counts = np.random.poisson(flux)

    per, st = ffa_search(counts, dt, pmin, pmax)
    #  fit_sinc wants frequencies, not periods
    model = fit(1/per[::-1], st[::-1], 1/period, obs_length=10)
    assert np.isclose(1/model.mean, period, atol=1e-6)


# @pytest.mark.skipif('not HAS_NUMBA')
def test_ffa_large_intv():
    period = 0.01
    pmin = 0.002789345
    pmax = 0.0105
    dt = 10**int(np.log10(0.002789345)) / 20
    length = 5
    times = np.arange(0, length, dt)

    flux = 10 + np.cos(2 * np.pi * times / period)

    counts = np.random.poisson(flux)

    per, st = ffa_search(counts, dt, pmin, pmax)
    #  fit_sinc wants frequencies, not periods
    model = fit(1/per[::-1], st[::-1], 1/period, obs_length=10)
    assert np.isclose(1/model.mean, period, atol=1e-6)
