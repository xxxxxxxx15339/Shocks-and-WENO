# -*- coding: utf-8 -*-
"""Deterministic and neural-selector TENO5 reconstruction schemes."""

import numpy as np

from ..core.SimulationClasses import FiniteVolumeMethod, FiniteVolumeMethodEuler
from .teno_common import combine_teno_candidates, normalize_teno_stencils


STENCIL_SIZE = 5
LINEAR_WEIGHTS = np.array([1.0/10.0, 3.0/5.0, 3.0/10.0])


def _candidate_values(values):
    """Return the three TENO5 candidates along the final stencil axis."""
    candidate0 = (
        1.0/3.0*values[..., 0] - 7.0/6.0*values[..., 1]
        + 11.0/6.0*values[..., 2]
    )
    candidate1 = (
        -1.0/6.0*values[..., 1] + 5.0/6.0*values[..., 2]
        + 1.0/3.0*values[..., 3]
    )
    candidate2 = (
        1.0/3.0*values[..., 2] + 5.0/6.0*values[..., 3]
        - 1.0/6.0*values[..., 4]
    )
    return np.stack([candidate0, candidate1, candidate2], axis=-1)


def _deterministic_keep(values):
    epsilon, power, cutoff = 1.0e-40, 6.0, 1.0e-6
    beta0 = (
        13.0/12.0*(values[..., 0]-2.0*values[..., 1]+values[..., 2])**2
        + 1.0/4.0*(values[..., 0]-4.0*values[..., 1]+3.0*values[..., 2])**2
    )
    beta1 = (
        13.0/12.0*(values[..., 1]-2.0*values[..., 2]+values[..., 3])**2
        + 1.0/4.0*(values[..., 1]-values[..., 3])**2
    )
    beta2 = (
        13.0/12.0*(values[..., 2]-2.0*values[..., 3]+values[..., 4])**2
        + 1.0/4.0*(3.0*values[..., 2]-4.0*values[..., 3]+values[..., 4])**2
    )
    beta = np.stack([beta0, beta1, beta2], axis=-1)
    log_gamma = -power*np.log(beta + epsilon)
    log_gamma -= np.max(log_gamma, axis=-1, keepdims=True)
    gamma = np.exp(log_gamma)
    return gamma/np.sum(gamma, axis=-1, keepdims=True) >= cutoff


def _neural_keep(model, values, threshold):
    normalized = normalize_teno_stencils(values)
    flat = normalized.reshape((-1, STENCIL_SIZE))
    probabilities = np.asarray(model.predict(flat, verbose=0))
    expected = (flat.shape[0], 3)
    if probabilities.shape != expected:
        raise ValueError(
            'NN-TENO5 model returned {}; expected {}.'.format(
                probabilities.shape, expected,
            )
        )
    return np.logical_not(
        probabilities.reshape(values.shape[:-1] + (3,)) >= threshold
    )


def _reconstruct(values):
    return combine_teno_candidates(
        _candidate_values(values), _deterministic_keep(values),
        LINEAR_WEIGHTS, values[..., 2],
    )


def TENO5():
    """Build deterministic scalar TENO5 reconstruction."""
    return FiniteVolumeMethod(STENCIL_SIZE, _reconstruct)


def TENO5euler():
    """Build deterministic characteristic TENO5 reconstruction."""
    return FiniteVolumeMethodEuler(STENCIL_SIZE, _reconstruct)


def NNMethod(model, threshold=0.5):
    """Build scalar NN-TENO5 from a troubled-stencil classifier."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('threshold must belong to [0, 1].')

    def scheme(u):
        values = np.asarray(u)
        if values.ndim != 2 or values.shape[1] != STENCIL_SIZE:
            raise ValueError('NN-TENO5 expects input shape (nx, 5).')
        candidates = _candidate_values(values)
        return combine_teno_candidates(
            candidates, _neural_keep(model, values, threshold),
            LINEAR_WEIGHTS, values[:, 2],
        )

    return FiniteVolumeMethod(STENCIL_SIZE, scheme)


def NNEuler(model, threshold=0.5):
    """Build characteristic NN-TENO5 using one batched model call."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('threshold must belong to [0, 1].')

    def scheme(f):
        values = np.asarray(f)
        if values.ndim != 3 or values.shape[2] != STENCIL_SIZE:
            raise ValueError('NN-TENO5 Euler expects shape (nx, nchar, 5).')
        candidates = _candidate_values(values)
        return combine_teno_candidates(
            candidates, _neural_keep(model, values, threshold),
            LINEAR_WEIGHTS, values[:, :, 2],
        )

    return FiniteVolumeMethodEuler(STENCIL_SIZE, scheme)
