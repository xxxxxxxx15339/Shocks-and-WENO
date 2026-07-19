from __future__ import print_function

import json

from src.analysis import (
    exact_sod_solution, interpolate_reference, regression_metrics,
    run_euler_benchmark,
)
from src.initial_conditions.InitialConditions import shuOsher, sod
from src.schemes import WENO3euler, WENO5euler, WENO7euler


def main():
    schemes = (WENO3euler, WENO5euler, WENO7euler)
    results = {'reference': {
        'sod': 'exact Riemann solution',
        'shu_osher': 'WENO7, 320 cells, CFL 0.35',
    }, 'sod': {}, 'shu_osher': {}}
    for scheme in schemes:
        x, solution, initial = run_euler_benchmark(
            scheme, sod(), 120, 1.0, 0.2
        )
        results['sod'][scheme.__name__] = regression_metrics(
            x, solution, initial, 0.2, exact_sod_solution(x, 0.2)
        )
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
