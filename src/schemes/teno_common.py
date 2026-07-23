import numpy as np


def normalize_teno_stencils(stencils):
    """Normalize each stencil along its final axis without modifying it."""
    scale = np.maximum(
        np.max(np.abs(stencils), axis=-1, keepdims=True),
        1.0,
    )
    return stencils/scale


def combine_teno_candidates(candidates, keep_mask, linear_weights, fallback):
    """Combine retained candidates and use fallback if all are rejected."""
    candidates = np.asarray(candidates, dtype=float)
    keep_mask = np.asarray(keep_mask, dtype=float)
    linear_weights = np.asarray(linear_weights, dtype=float)

    active_weights = keep_mask*linear_weights
    denominator = np.sum(active_weights, axis=-1)
    numerator = np.sum(active_weights*candidates, axis=-1)
    result = np.array(fallback, dtype=float, copy=True)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result


# Compatibility with the name used by the first deterministic implementation.
teno_common = combine_teno_candidates
