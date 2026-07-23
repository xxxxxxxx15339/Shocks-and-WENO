# -*- coding: utf-8 -*-
"""Deterministic and neural-selector TENO3 reconstruction schemes."""

import numpy as np

from ..core.SimulationClasses import FiniteVolumeMethod, FiniteVolumeMethodEuler
from .teno_common import combine_teno_candidates, normalize_teno_stencils


STENCIL_SIZE = 3
LINEAR_WEIGHTS = np.array([1.0/3.0, 2.0/3.0])


def _candidate_values(values):
    """Return the two TENO3 candidates along the final stencil axis."""
    candidate0 = -0.5*values[..., 0] + 1.5*values[..., 1]
    candidate1 = 0.5*values[..., 1] + 0.5*values[..., 2]
    return np.stack([candidate0, candidate1], axis=-1)


def _deterministic_keep(values):
    epsilon, power, cutoff = 1.0e-40, 6.0, 1.0e-6
    beta0 = (values[..., 1] - values[..., 0])**2
    beta1 = (values[..., 2] - values[..., 1])**2
    log_gamma = -power*np.log(
        np.stack([beta0, beta1], axis=-1) + epsilon
    )
    log_gamma -= np.max(log_gamma, axis=-1, keepdims=True)
    gamma = np.exp(log_gamma)
    chi = gamma/np.sum(gamma, axis=-1, keepdims=True)
    return chi >= cutoff


def _neural_keep(model, values, threshold):
    normalized = normalize_teno_stencils(values)
    flat = normalized.reshape((-1, STENCIL_SIZE))
    probabilities = np.asarray(model.predict(flat, verbose=0))
    expected = (flat.shape[0], 2)
    if probabilities.shape != expected:
        raise ValueError(
            'NN-TENO3 model returned {}; expected {}.'.format(
                probabilities.shape, expected,
            )
        )
    return np.logical_not(
        probabilities.reshape(values.shape[:-1] + (2,)) >= threshold
    )


def _reconstruct(values):
    candidates = _candidate_values(values)
    return combine_teno_candidates(
        candidates,
        _deterministic_keep(values),
        LINEAR_WEIGHTS,
        values[..., 1],
    )


def TENO3():
    """Build deterministic scalar TENO3 reconstruction."""
    return FiniteVolumeMethod(STENCIL_SIZE, _reconstruct)


def TENO3euler():
    """Build deterministic characteristic TENO3 reconstruction."""
    return FiniteVolumeMethodEuler(STENCIL_SIZE, _reconstruct)


def NNMethod(model, threshold=0.5):
    """Build scalar NN-TENO3 from a troubled-stencil classifier."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('threshold must belong to [0, 1].')

    def scheme(u):
        values = np.asarray(u)
        if values.ndim != 2 or values.shape[1] != STENCIL_SIZE:
            raise ValueError('NN-TENO3 expects input shape (nx, 3).')
        candidates = _candidate_values(values)
        return combine_teno_candidates(
            candidates, _neural_keep(model, values, threshold),
            LINEAR_WEIGHTS, values[:, 1],
        )

    return FiniteVolumeMethod(STENCIL_SIZE, scheme)


def NNEuler(model, threshold=0.5):
    """Build characteristic NN-TENO3 using one batched model call."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('threshold must belong to [0, 1].')

    def scheme(f):
        values = np.asarray(f)
        if values.ndim != 3 or values.shape[2] != STENCIL_SIZE:
            raise ValueError('NN-TENO3 Euler expects shape (nx, nchar, 3).')
        candidates = _candidate_values(values)
        return combine_teno_candidates(
            candidates, _neural_keep(model, values, threshold),
            LINEAR_WEIGHTS, values[:, :, 1],
        )

    return FiniteVolumeMethodEuler(STENCIL_SIZE, scheme)
