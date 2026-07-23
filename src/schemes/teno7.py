# -*- coding: utf-8 -*-
"""Deterministic and neural-selector TENO7 reconstruction schemes."""

import numpy as np

from ..core.SimulationClasses import FiniteVolumeMethod, FiniteVolumeMethodEuler
from .teno_common import combine_teno_candidates, normalize_teno_stencils


STENCIL_SIZE = 7
LINEAR_WEIGHTS = np.array([
    1.0/35.0, 12.0/35.0, 18.0/35.0, 4.0/35.0,
])


def _candidate_values(values):
    """Return the four TENO7 candidates along the final stencil axis."""
    u0, u1, u2, u3, u4, u5, u6 = [values[..., i] for i in range(7)]
    candidate0 = -1.0/4.0*u0 + 13.0/12.0*u1 - 23.0/12.0*u2 + 25.0/12.0*u3
    candidate1 = 1.0/12.0*u1 - 5.0/12.0*u2 + 13.0/12.0*u3 + 1.0/4.0*u4
    candidate2 = -1.0/12.0*u2 + 7.0/12.0*u3 + 7.0/12.0*u4 - 1.0/12.0*u5
    candidate3 = 1.0/4.0*u3 + 13.0/12.0*u4 - 5.0/12.0*u5 + 1.0/12.0*u6
    return np.stack(
        [candidate0, candidate1, candidate2, candidate3],
        axis=-1,
    )


def _smoothness_indicators(values):
    u0, u1, u2, u3, u4, u5, u6 = [values[..., i] for i in range(7)]
    beta0 = (
        2107.0/240.0*u3**2 - 1567.0/40.0*u3*u2
        + 3521.0/120.0*u3*u1 - 309.0/40.0*u3*u0
        + 11003.0/240.0*u2**2 - 8623.0/120.0*u2*u1
        + 2321.0/120.0*u2*u0 + 7043.0/240.0*u1**2
        - 647.0/40.0*u1*u0 + 547.0/240.0*u0**2
    )
    beta1 = (
        3443.0/240.0*u3**2 - 1261.0/120.0*u3*u4
        - 2983.0/120.0*u3*u2 + 267.0/40.0*u3*u1
        + 547.0/240.0*u4**2 + 961.0/120.0*u4*u2
        - 247.0/120.0*u4*u1 + 2843.0/240.0*u2**2
        - 821.0/120.0*u2*u1 + 89.0/80.0*u1**2
    )
    beta2 = (
        3443.0/240.0*u3**2 - 2983.0/120.0*u3*u4
        + 267.0/40.0*u3*u5 - 1261.0/120.0*u3*u2
        + 2843.0/240.0*u4**2 - 821.0/120.0*u4*u5
        + 961.0/120.0*u4*u2 + 89.0/80.0*u5**2
        - 247.0/120.0*u5*u2 + 547.0/240.0*u2**2
    )
    beta3 = (
        2107.0/240.0*u3**2 - 1567.0/40.0*u3*u4
        + 3521.0/120.0*u3*u5 - 309.0/40.0*u3*u6
        + 11003.0/240.0*u4**2 - 8623.0/120.0*u4*u5
        + 2321.0/120.0*u4*u6 + 7043.0/240.0*u5**2
        - 647.0/40.0*u5*u6 + 547.0/240.0*u6**2
    )
    return beta0, beta1, beta2, beta3


def _deterministic_keep(values):
    epsilon, power, cutoff = 1.0e-40, 6.0, 1.0e-6
    beta = np.stack(_smoothness_indicators(values), axis=-1)
    log_gamma = -power*np.log(np.maximum(beta, 0.0) + epsilon)
    log_gamma -= np.max(log_gamma, axis=-1, keepdims=True)
    gamma = np.exp(log_gamma)
    return gamma/np.sum(gamma, axis=-1, keepdims=True) >= cutoff


def _neural_keep(model, values, threshold):
    normalized = normalize_teno_stencils(values)
    flat = normalized.reshape((-1, STENCIL_SIZE))
    probabilities = np.asarray(model.predict(flat, verbose=0))
    expected = (flat.shape[0], 4)
    if probabilities.shape != expected:
        raise ValueError(
            'NN-TENO7 model returned {}; expected {}.'.format(
                probabilities.shape, expected,
            )
        )
    return np.logical_not(
        probabilities.reshape(values.shape[:-1] + (4,)) >= threshold
    )


def _reconstruct(values):
    return combine_teno_candidates(
        _candidate_values(values), _deterministic_keep(values),
        LINEAR_WEIGHTS, values[..., 3],
    )


def TENO7():
    """Build deterministic scalar TENO7 reconstruction."""
    return FiniteVolumeMethod(STENCIL_SIZE, _reconstruct)


def TENO7euler():
    """Build deterministic characteristic TENO7 reconstruction."""
    return FiniteVolumeMethodEuler(STENCIL_SIZE, _reconstruct)


def NNMethod(model, threshold=0.5):
    """Build scalar NN-TENO7 from a troubled-stencil classifier."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('threshold must belong to [0, 1].')

    def scheme(u):
        values = np.asarray(u)
        if values.ndim != 2 or values.shape[1] != STENCIL_SIZE:
            raise ValueError('NN-TENO7 expects input shape (nx, 7).')
        candidates = _candidate_values(values)
        return combine_teno_candidates(
            candidates, _neural_keep(model, values, threshold),
            LINEAR_WEIGHTS, values[:, 3],
        )

    return FiniteVolumeMethod(STENCIL_SIZE, scheme)


def NNEuler(model, threshold=0.5):
    """Build characteristic NN-TENO7 using one batched model call."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('threshold must belong to [0, 1].')

    def scheme(f):
        values = np.asarray(f)
        if values.ndim != 3 or values.shape[2] != STENCIL_SIZE:
            raise ValueError('NN-TENO7 Euler expects shape (nx, nchar, 7).')
        candidates = _candidate_values(values)
        return combine_teno_candidates(
            candidates, _neural_keep(model, values, threshold),
            LINEAR_WEIGHTS, values[:, :, 3],
        )

    return FiniteVolumeMethodEuler(STENCIL_SIZE, scheme)
