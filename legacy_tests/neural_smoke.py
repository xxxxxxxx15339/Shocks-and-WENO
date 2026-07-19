import unittest

import numpy as np

from src.networks.wholeNetworks import WENO51stOrder


class NeuralPipelineSmokeTests(unittest.TestCase):
    def test_weno5_model_builds_and_predicts(self):
        model = WENO51stOrder(0.15)
        prediction = model.predict(np.zeros((2, 5)), verbose=0)
        self.assertEqual(prediction.shape, (2, 1))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == '__main__':
    unittest.main()
