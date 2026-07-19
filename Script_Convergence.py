from __future__ import print_function

from src.analysis import periodic_advection_convergence
from src.schemes import WENO3, WENO5, WENO7


def main():
    studies = (
        ('WENO3', WENO3, (320, 640, 1280)),
        ('WENO5', WENO5, (80, 160, 320)),
        ('WENO7', WENO7, (160, 320, 640)),
    )
    for name, builder, resolutions in studies:
        print(name)
        print('cells       L1 error       observed order')
        for row in periodic_advection_convergence(builder, resolutions):
            order = '-' if row['order'] is None else '{:.3f}'.format(row['order'])
            print('{:5d}   {:13.6e}   {:>14}'.format(
                row['resolution'], row['l1_error'], order
            ))
        print('')


if __name__ == '__main__':
    main()
