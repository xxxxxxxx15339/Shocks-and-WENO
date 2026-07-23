# -*- coding: utf-8 -*-
"""Generate characteristic Euler WENO5-NN training data.

The labels are WENO7 reconstructions of Lax--Friedrichs split characteristic
fluxes.  The input is the matching central five-point stencil, which is the
input expected by ``NNEuler5``.
"""

import argparse
import os

import numpy as np

from src.core.SimulationClasses import eulerSimulation, FiniteVolumeMethodEuler
from src.core.TimeSteppingMethods import SSPRK3
from src.core.eulerEquations import (
    flux, getEulerFlux, project_to_characteristic, roe_eigenbasis, spds,
)
from src.initial_conditions.InitialConditions import shuOsher, sod
from src.schemes import WENO7euler


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Euler NN CSV data.')
    parser.add_argument('--output-inputs', default='data/eulerAvgs.csv')
    parser.add_argument('--output-targets', default='data/eulerFlux.csv')
    parser.add_argument('--output-groups', default='data/eulerGroups.csv')
    parser.add_argument('--samples', type=int, default=100000,
                        help='Total number of generated samples.')
    parser.add_argument('--cells', type=int, default=256)
    parser.add_argument('--cfl', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def smooth_wave(x, frequency, phase, amplitude):
    rho = 1.0 + amplitude*np.sin(frequency*(x-5.0)+phase)
    velocity = 0.15*np.cos(frequency*(x-5.0)+phase)
    pressure = 1.0 + 0.1*amplitude*np.sin(frequency*(x-5.0)+phase)
    energy = pressure/0.4 + 0.5*rho*velocity**2
    return np.column_stack((rho, rho*velocity, energy))


def trajectories(args, rng):
    x = np.arange(args.cells, dtype=float)*10.0/args.cells
    cases = []
    for frequency in (0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0,
                      9.0, 11.0):
        for phase in (0.0, 0.7, 1.3, 2.1):
            cases.append(('smooth', smooth_wave(
                x, frequency, phase, rng.uniform(0.05, 0.25)
            )))
    for name, initial_condition in (('shock', sod()), ('shock_oscillation', shuOsher())):
        sim = eulerSimulation(
            args.cells, 2, 10.0, 1.8, SSPRK3(),
            getEulerFlux(__import__('src.schemes', fromlist=['WENO7euler']).WENO7euler()),
            initial_condition, 3, max_cfl=args.cfl,
            wave_speed_function=lambda state: np.max(np.abs(spds(state))),
        )
        history = np.transpose(sim.runEuler(), (1, 0, 2))
        for state in history[::max(1, len(history)//24)]:
            cases.append((name, state))
    for _ in range(12):
        rho = rng.uniform(0.5, 3.0)
        velocity = rng.uniform(-1.0, 2.5)
        pressure = rng.uniform(0.2, 12.0)
        energy = pressure/0.4 + 0.5*rho*velocity**2
        cases.append(('constant', np.tile(
            [rho, rho*velocity, energy], (args.cells, 1)
        )))
    return cases


def split_characteristic_stencils(state):
    """Return five-point split stencils and WENO7 targets per scalar field."""
    cells = len(state)
    f = flux(state)
    speeds = np.max(np.abs(spds(state)), axis=1)
    right_state = np.concatenate((state[1:], state[-1:]), axis=0)
    right_speed = np.concatenate((speeds[1:], speeds[-1:]))
    alpha = np.maximum(speeds, right_speed)
    left, right = roe_eigenbasis(state)

    def stencil(values, offset, reverse=False):
        indices = np.clip(
            np.arange(cells)[:, None] + offset
            + np.arange(-3, 4)[None, :], 0, cells-1
        )
        result = np.transpose(values[indices], (0, 2, 1))
        return np.flip(result, axis=2) if reverse else result

    up = project_to_characteristic(stencil(state, 0), left)
    fp = project_to_characteristic(stencil(f, 0), left)
    um = project_to_characteristic(stencil(state, 1, True), left)
    fm = project_to_characteristic(stencil(f, 1, True), left)
    pos = 0.5*(fp + alpha[:, None, None]*up)
    neg = 0.5*(fm - alpha[:, None, None]*um)

    weno7 = __import__('src.schemes', fromlist=['WENO7euler']).WENO7euler()
    inputs = np.concatenate((pos[:, :, 1:6], neg[:, :, 1:6]), axis=0)
    targets = np.concatenate((weno7.evalF(pos), weno7.evalF(neg)), axis=0)
    # Flatten characteristic components into independent scalar training rows.
    return inputs.transpose(0, 1, 2).reshape(-1, 5), targets.reshape(-1)


def main():
    args = parse_args()
    if args.samples < 1:
        raise ValueError('--samples must be positive.')
    rng = np.random.RandomState(args.seed)
    candidates = {name: [] for name in (
        'smooth', 'shock', 'shock_oscillation', 'constant'
    )}
    for name, state in trajectories(args, rng):
        inputs, targets = split_characteristic_stencils(state)
        candidates[name].append((inputs, targets))

    # The requested mixture is sampled without allowing one trajectory to
    # dominate the dataset.
    proportions = {
        'smooth': 0.40, 'shock': 0.30,
        'shock_oscillation': 0.20, 'constant': 0.10,
    }
    all_inputs, all_targets, all_groups = [], [], []
    for group, fraction in proportions.items():
        inputs = np.concatenate([item[0] for item in candidates[group]], axis=0)
        targets = np.concatenate([item[1] for item in candidates[group]], axis=0)
        count = min(len(inputs), max(1, int(args.samples*fraction)))
        # Select the desired group size, preserving stochastic variety.
        selected = rng.choice(len(inputs), size=count, replace=False)
        all_inputs.append(inputs[selected])
        all_targets.append(targets[selected])
        all_groups.extend([group]*len(selected))

    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    groups = np.asarray(all_groups)
    order = rng.permutation(len(inputs))
    inputs, targets, groups = inputs[order], targets[order], groups[order]
    for path in (args.output_inputs, args.output_targets, args.output_groups):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    np.savetxt(args.output_inputs, inputs.T, delimiter=',')
    np.savetxt(args.output_targets, targets[:, None], delimiter=',')
    np.savetxt(args.output_groups, groups[:, None], delimiter=',', fmt='%s')
    print('Wrote {} samples to {}'.format(len(inputs), args.output_inputs))
    print('Targets: {}'.format(args.output_targets))
    print('Groups: {}'.format(args.output_groups))


if __name__ == '__main__':
    main()
