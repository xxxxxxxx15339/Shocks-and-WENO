import unittest

import numpy as np

from Script_TrainNetworksFixLeak import scale_training_data, split_dataset
from src.core.Equations import adv
from src.core.FluxSplittingMethods import LaxFriedrichs
from src.core.SimulationClasses import Simulation, eulerSimulation
from src.core.TimeSteppingMethods import SSPRK3
from src.initial_conditions.InitialConditions import sod, step1
from src.schemes import WENO5


class WENO5Tests(unittest.TestCase):
    def test_constant_reconstruction(self):
        values = np.full(32, 2.5)
        np.testing.assert_allclose(WENO5().evalF(values), values)

    def test_short_periodic_advection_is_finite(self):
        simulation = Simulation(
            40, 11, 2.0, 0.02, SSPRK3(), LaxFriedrichs(adv(), 1),
            WENO5(), step1(), max_cfl=1.0,
        )
        solution = simulation.run()
        self.assertTrue(np.isfinite(solution).all())

    def test_cfl_violation_is_rejected(self):
        simulation = Simulation(
            20, 2, 2.0, 1.0, SSPRK3(), LaxFriedrichs(adv(), 1),
            WENO5(), step1(), max_cfl=1.0,
        )
        with self.assertRaisesRegex(ValueError, 'CFL'):
            simulation.run()

    def test_unsupported_scalar_boundary_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            WENO5().__class__(5, lambda values: values, boundary='edge')


class DataPipelineTests(unittest.TestCase):
    def test_scaling_round_trip_and_constants(self):
        inputs = np.array([[2., 4., 6.], [3., 3., 3.]])
        targets = np.array([[5.], [3.]])
        scaled_inputs, scaled_targets = scale_training_data(inputs, targets)
        np.testing.assert_allclose(scaled_inputs[0], [0., 0.5, 1.])
        np.testing.assert_allclose(scaled_targets[0], [0.75])
        np.testing.assert_allclose(scaled_inputs[1], 0.)
        np.testing.assert_allclose(scaled_targets[1], 0.)

    def test_contiguous_train_validation_test_split(self):
        inputs = np.arange(100).reshape(20, 5)
        targets = np.arange(20).reshape(20, 1)
        x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(
            inputs, targets
        )
        self.assertEqual((len(x_train), len(x_val), len(x_test)), (14, 3, 3))
        self.assertLess(y_train[-1, 0], y_val[0, 0])
        self.assertLess(y_val[-1, 0], y_test[0, 0])


class EulerValidationTests(unittest.TestCase):
    def test_nonpositive_density_is_rejected(self):
        simulation = eulerSimulation(
            5, 2, 1.0, 0.01, SSPRK3(), lambda state: state,
            sod(), 3,
        )
        invalid = np.ones((5, 3))
        invalid[0, 0] = 0
        with self.assertRaisesRegex(FloatingPointError, 'density'):
            simulation._validate_state(invalid, 0)


if __name__ == '__main__':
    unittest.main()
