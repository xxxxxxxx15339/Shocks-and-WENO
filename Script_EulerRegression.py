from __future__ import print_function

import argparse
import json

import numpy as np

from src.analysis import (
    exact_sod_solution, interpolate_reference, regression_metrics,
    run_euler_benchmark,
)
from src.initial_conditions.InitialConditions import shuOsher, sod
from src.schemes import WENO3euler, WENO5euler, WENO7euler


def parse_args():
    parser = argparse.ArgumentParser(description='Run Euler regression metrics.')
    parser.add_argument(
        '--shu-reference',
        help='Independent CSV with columns x,density,momentum,energy.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    schemes = (WENO3euler, WENO5euler, WENO7euler)
    results = {'reference': {
        'sod': 'exact Riemann solution',
        'shu_osher': 'internal regression baseline: WENO7, 320 cells, CFL 0.35',
    }, 'sod': {}, 'shu_osher': {}}
    for scheme in schemes:
        x, solution, initial = run_euler_benchmark(
            scheme, sod(), 120, 1.0, 0.2
        )
        results['sod'][scheme.__name__] = regression_metrics(
            x, solution, initial, 0.2, exact_sod_solution(x, 0.2)
        )
    if args.shu_reference:
        supplied = np.loadtxt(args.shu_reference, delimiter=',')
        if supplied.ndim != 2 or supplied.shape[1] != 4:
            raise ValueError('Shu--Osher reference must have four CSV columns.')
        reference_x, reference = supplied[:,0], supplied[:,1:]
        results['reference']['shu_osher'] = args.shu_reference
    else:
        reference_x, reference, _ = run_euler_benchmark(
            WENO7euler, shuOsher(), 320, 10.0, 1.8
        )
    for scheme in schemes:
        x, solution, initial = run_euler_benchmark(
            scheme, shuOsher(), 120, 10.0, 1.8
        )
        results['shu_osher'][scheme.__name__] = regression_metrics(
            x, solution, initial, 1.8,
            interpolate_reference(reference_x, reference, x),
        )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
