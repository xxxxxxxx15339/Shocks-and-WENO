# -*- coding: utf-8 -*-
"""
Neural candidate-stencil classifiers for NN-TENO3, NN-TENO5, and NN-TENO7.

The architecture follows the multilayer perceptron used by the NENO6 method.
Each output is an independent probability that one candidate stencil is
troubled:

    0 = smooth candidate (keep)
    1 = troubled candidate (discard)

This module builds classifiers only.  Candidate polynomials, optimal linear
weights, stencil selection, fallback fluxes, and final reconstruction belong
to the TENO scheme wrappers.
"""

from keras import optimizers
from keras.layers import Dense, Input
from keras.models import Model


def build_teno_network(stencil_size, number_of_candidates,
                       learning_rate=1.0e-4):
    """Build and compile a NENO-style multi-label stencil classifier.

    Parameters
    ----------
    stencil_size : int
        Number of values in each normalized input stencil.
    number_of_candidates : int
        Number of candidate-stencil troubled probabilities to predict.
    learning_rate : float, optional
        Adam learning rate.  The NENO6 reference uses ``1e-4`` initially.

    Returns
    -------
    keras.models.Model
        Compiled classifier with binary-cross-entropy loss.
    """
    if not isinstance(stencil_size, int) or stencil_size <= 0:
        raise ValueError('stencil_size must be a positive integer.')
    if not isinstance(number_of_candidates, int) or number_of_candidates <= 0:
        raise ValueError('number_of_candidates must be a positive integer.')
    if learning_rate <= 0.0:
        raise ValueError('learning_rate must be positive.')

    stencil = Input(
        shape=(stencil_size,),
        name='teno_stencil',
    )

    hidden1 = Dense(64, activation='relu', name='hidden_64')(stencil)
    hidden2 = Dense(32, activation='relu', name='hidden_32')(hidden1)
    hidden3 = Dense(16, activation='relu', name='hidden_16')(hidden2)
    hidden4 = Dense(8, activation='relu', name='hidden_8')(hidden3)

    troubled_probabilities = Dense(
        number_of_candidates,
        activation='sigmoid',
        name='troubled_probabilities',
    )(hidden4)

    model = Model(
        inputs=stencil,
        outputs=troubled_probabilities,
        name='nn_teno{}_classifier'.format(stencil_size),
    )

    # The repository targets standalone Keras 2.2.4, where Adam uses ``lr``.
    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
    )

    return model


def TENO3Network(learning_rate=1.0e-4):
    """Build NN-TENO3: ``3 -> 64 -> 32 -> 16 -> 8 -> 2``."""
    return build_teno_network(
        stencil_size=3,
        number_of_candidates=2,
        learning_rate=learning_rate,
    )


def TENO5Network(learning_rate=1.0e-4):
    """Build NN-TENO5: ``5 -> 64 -> 32 -> 16 -> 8 -> 3``."""
    return build_teno_network(
        stencil_size=5,
        number_of_candidates=3,
        learning_rate=learning_rate,
    )


def TENO7Network(learning_rate=1.0e-4):
    """Build NN-TENO7: ``7 -> 64 -> 32 -> 16 -> 8 -> 4``."""
    return build_teno_network(
        stencil_size=7,
        number_of_candidates=4,
        learning_rate=learning_rate,
    )


__all__ = [
    'build_teno_network',
    'TENO3Network', 'TENO5Network', 'TENO7Network',
]
