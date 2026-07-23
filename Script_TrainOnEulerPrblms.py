# -*- coding: utf-8 -*-
"""Train the WENO5-NN reconstruction on generated Euler data."""

import argparse
import datetime
import os
import random

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from keras import optimizers
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping

from src.analysis.euler_regression import (
    exact_sod_solution, interpolate_reference, primitive_variables,
    regression_metrics,
)
from src.data.preprocessing import scale_training_data
from src.networks.WenoNetworks import WENO51stOrder
from src.initial_conditions.InitialConditions import shuOsher, sod
from src.schemes import NNEuler5, WENO5euler, WENO7euler
from Script_Euler_Sod import run_history


class CompactTrainingLogger(Callback):
    """Keep one compact line per epoch, matching the scalar trainer."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(
            '  Epoch {:03d} | loss {:.6f} | val_loss {:.6f}'.format(
                epoch + 1,
                logs.get('loss', float('nan')),
                logs.get('val_loss', float('nan')),
            ),
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train WENO5-NN using Euler characteristic stencils.'
    )
    parser.add_argument('--input-data', default='data/eulerAvgs.csv')
    parser.add_argument('--target-data', default='data/eulerFlux.csv')
    parser.add_argument('--model-path', default='trained_euler_weno5.h5')
    parser.add_argument('--regularization', type=float, default=0.10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-attempts', type=int, default=100)
    parser.add_argument('--cells', type=int, default=120)
    parser.add_argument('--reference-cells', type=int, default=320)
    parser.add_argument('--cfl', type=float, default=0.35)
    return parser.parse_args()


def load_csv_data(input_path, target_path):
    inputs = np.loadtxt(input_path, delimiter=',')
    targets = np.loadtxt(target_path, delimiter=',').reshape(-1, 1)
    if inputs.ndim != 2 or inputs.shape[0] != 5:
        raise ValueError('Euler input data must have shape (5, samples).')
    inputs = inputs.T
    if len(inputs) != len(targets):
        raise ValueError('Euler input and target sample counts differ.')
    return inputs.astype(np.float32), targets.astype(np.float32)


def run_final(scheme_builder, initial_condition, cells, length, final_time,
              cfl):
    return run_history(
        scheme_builder, initial_condition, cells, length, final_time, cfl,
    )


def l1_error(candidate, reference):
    return float(np.mean(np.abs(candidate-reference)))


def density_metrics(history):
    """Return maximum density TV and local 10--90% shock width in cells."""
    density = history[:, :, 0]
    total_variation = np.sum(np.abs(np.diff(density, axis=1)), axis=1)
    gradients = np.abs(np.diff(density, axis=1))
    widths = []
    for state, row in zip(density, gradients):
        center = int(np.argmax(row))
        # Estimate the two local plateaus around the strongest shock.  The
        # window is deliberately local so Shu--Osher waves are not counted.
        radius = min(8, len(state)//4)
        left_state = state[max(0, center-radius)]
        right_state = state[min(len(state)-1, center+radius)]
        jump = abs(right_state-left_state)
        if jump == 0:
            widths.append(0)
            continue
        lower = min(left_state, right_state) + 0.10*jump
        upper = max(left_state, right_state) - 0.10*jump
        lo = max(0, center-radius)
        hi = min(len(state), center+radius+2)
        transition = (state[lo:hi] >= lower) & (state[lo:hi] <= upper)
        center_local = center-lo
        if center_local >= len(transition) or not transition[center_local]:
            widths.append(1)
            continue
        left = center_local
        right = center_local
        while left > 0 and transition[left-1]:
            left -= 1
        while right+1 < len(transition) and transition[right+1]:
            right += 1
        widths.append(right-left+1)
    return float(np.max(total_variation)), int(np.max(widths))


def make_references(args):
    """Compute analytical Sod and high-resolution Shu--Osher references."""
    sod_x = np.linspace(0.0, 1.0, args.reference_cells, endpoint=False)
    sod_ref = exact_sod_solution(sod_x, 0.2)
    shu_x, _, shu_history = run_final(
        WENO7euler, shuOsher(), args.reference_cells, 10.0, 1.8, args.cfl
    )
    return sod_x, sod_ref, shu_x, shu_history[-1]


def evaluate_candidate(model, args, references):
    """Evaluate the exact Sod/Shu acceptance conditions without saving."""
    sod_ref_x, sod_ref, shu_ref_x, shu_ref = references
    sod_x, _, sod_weno_history = run_final(
        WENO5euler, sod(), args.cells, 1.0, 0.2, args.cfl
    )
    shu_x, _, shu_weno_history = run_final(
        WENO5euler, shuOsher(), args.cells, 10.0, 1.8, args.cfl
    )
    _, _, sod_nn_history = run_final(
        lambda: NNEuler5(model), sod(), args.cells, 1.0, 0.2, args.cfl
    )
    _, _, shu_nn_history = run_final(
        lambda: NNEuler5(model), shuOsher(), args.cells, 10.0, 1.8, args.cfl
    )
    sod_reference = interpolate_reference(sod_ref_x, sod_ref, sod_x)
    shu_reference = interpolate_reference(shu_ref_x, shu_ref, shu_x)
    sod_initial = sod()(sod_x)
    shu_initial = shuOsher()(shu_x)
    sod_weno = regression_metrics(
        sod_x, sod_weno_history[-1], sod_initial, 0.2, sod_reference
    )
    sod_nn = regression_metrics(
        sod_x, sod_nn_history[-1], sod_initial, 0.2, sod_reference
    )
    shu_weno = regression_metrics(
        shu_x, shu_weno_history[-1], shu_initial, 1.8, shu_reference
    )
    shu_nn = regression_metrics(
        shu_x, shu_nn_history[-1], shu_initial, 1.8, shu_reference
    )
    # Compare the post-shock oscillatory region, not the shock itself.
    regional = shu_x >= shu_weno['shock_location']
    shu_regional_l1_weno = float(np.mean(np.abs(
        shu_weno_history[-1][regional, 0]-shu_reference[regional, 0]
    )))
    shu_regional_l1_nn = float(np.mean(np.abs(
        shu_nn_history[-1][regional, 0]-shu_reference[regional, 0]
    )))
    physical_pass = True
    for history in (sod_nn_history, shu_nn_history):
        states = history.reshape(-1, 3)
        density, _, pressure = primitive_variables(states)
        physical_pass = physical_pass and (
            np.isfinite(states).all()
            and np.all(density > 0)
            and np.all(pressure > 0)
        )
    accepted = (
        physical_pass
        and sod_nn['density_l1'] < 0.012
        and sod_nn['momentum_l1'] < 0.012
        and sod_nn['energy_l1'] < 0.025
        and sod_nn['minimum_density'] > 0.12
        and sod_nn['minimum_pressure'] > 0.095
        and sod_nn['shock_location_error'] < 0.02
        and sod_nn['contact_location_error'] < 0.03
        and shu_nn['density_l1'] < 0.16
        and shu_nn['momentum_l1'] < 0.25
        and shu_nn['energy_l1'] < 0.95
        and shu_nn['minimum_density'] > 0.75
        and shu_nn['minimum_pressure'] > 0.99
        and shu_nn['shock_location_error'] < 0.15
        and shu_regional_l1_nn <= 0.98*shu_regional_l1_weno
    )
    return {
        'sod': sod_nn, 'shu': shu_nn,
        'sod_weno': sod_weno, 'shu_weno': shu_weno,
        'shu_regional_l1_nn': shu_regional_l1_nn,
        'shu_regional_l1_weno': shu_regional_l1_weno,
        'physical_pass': physical_pass, 'accepted': accepted,
    }


def main():
    args = parse_args()
    if args.max_attempts < 1:
        raise ValueError('--max-attempts must be positive.')
    tf.logging.set_verbosity(tf.logging.ERROR)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    inputs, targets = load_csv_data(args.input_data, args.target_data)
    # NNEuler5 performs this same per-stencil normalization at inference time.
    inputs, targets = scale_training_data(inputs, targets)
    order = np.random.RandomState(args.seed).permutation(len(inputs))
    inputs, targets = inputs[order], targets[order]
    train_end = int(0.70*len(inputs))
    validation_end = int(0.85*len(inputs))
    x_train, y_train = inputs[:train_end], targets[:train_end]
    x_validation, y_validation = (
        inputs[train_end:validation_end], targets[train_end:validation_end]
    )
    x_test, y_test = inputs[validation_end:], targets[validation_end:]

    print(
        'Training Euler WENO5-NN with train/validation/test sizes {}/{}/{}.'
        .format(len(x_train), len(x_validation), len(x_test)),
        flush=True,
    )

    print('Computing Sod exact and fixed WENO7 Shu--Osher references...')
    references = make_references(args)
    best = None
    for attempt in range(1, args.max_attempts+1):
        started = datetime.datetime.now()
        attempt_seed = args.seed + attempt - 1
        random.seed(attempt_seed)
        np.random.seed(attempt_seed)
        tf.set_random_seed(attempt_seed)
        print('\n=== Euler attempt {} ==='.format(attempt))
        model = WENO51stOrder(args.regularization)
        model.compile(
            optimizer=optimizers.adam(lr=args.learning_rate),
            loss='mean_squared_error',
        )
        stopping = EarlyStopping(
            monitor='val_loss', patience=5, min_delta=1e-5,
            restore_best_weights=True,
        )
        compact_logger = CompactTrainingLogger()
        model.fit(
            x_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_validation, y_validation),
            callbacks=[stopping, compact_logger],
            verbose=0,
        )
        test_loss = float(model.evaluate(x_test, y_test, verbose=0))
        try:
            metrics = evaluate_candidate(model, args, references)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
            print('Candidate rejected during Euler evaluation: {}'.format(error))
            K.clear_session()
            continue
        metrics['test_loss'] = test_loss
        print('  Sod density/momentum/energy L1: {:.6e} / {:.6e} / {:.6e}'.format(
            metrics['sod']['density_l1'], metrics['sod']['momentum_l1'],
            metrics['sod']['energy_l1']))
        print('  Shu density/momentum/energy L1: {:.6e} / {:.6e} / {:.6e}'.format(
            metrics['shu']['density_l1'], metrics['shu']['momentum_l1'],
            metrics['shu']['energy_l1']))
        print('  Shu regional NN/WENO5 L1: {:.6e} / {:.6e}'.format(
            metrics['shu_regional_l1_nn'], metrics['shu_regional_l1_weno']))
        print('  Duration  | {}'.format(datetime.datetime.now() - started),
              flush=True)
        print('  Test loss | {:.8e}'.format(test_loss), flush=True)
        print('  Accepted: {}'.format(metrics['accepted']))
        score = (
            max(0.0, metrics['sod']['density_l1']-0.012),
            max(0.0, metrics['shu']['density_l1']-0.16),
            max(0.0, metrics['shu_regional_l1_nn']-
                0.98*metrics['shu_regional_l1_weno']),
            test_loss,
        )
        if best is None or score < best['score']:
            best = {'score': score, 'metrics': metrics}
        if metrics['accepted']:
            model.save(args.model_path)
            print('Saved accepted Euler model: {}'.format(args.model_path))
            return
        K.clear_session()

    print('No Euler candidate met the acceptance conditions.')
    print('Best candidate metrics: {}'.format(best))
    print('No model was saved.')
    raise SystemExit(1)


if __name__ == '__main__':
    main()
