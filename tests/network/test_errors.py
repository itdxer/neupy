import numpy as np

from neupy.network import errors

from base import BaseTestCase


class ErrorFuncTestCase(BaseTestCase):
    def test_mse(self):
        predicted = np.array([
            [1, 2],
            [3, 4],
            [-5, -6]
        ])
        target = np.array([
            [1, 1],
            [1, 1],
            [1, 1]
        ])

        actual = errors.mse(predicted, target)
        expected = 16.5

        self.assertAlmostEqual(actual.eval(), expected)

    def test_binary_crossentropy(self):
        predicted = np.array([
            [0.4],
            [0],
            [0.01],
        ])
        target = np.array([
            [1],
            [0],
            [0],
        ])

        actual = errors.binary_crossentropy(predicted, target)
        expected = 4.68

        self.assertAlmostEqual(actual.eval(), expected, places=2)

    def test_categorical_crossentropy(self):
        predicted = np.array([
            [0.4, 0.6],
            [0, 1],
            [0.01, 0.99],
        ])
        target = np.array([
            [1, 0],
            [0, 1],
            [0, 1],
        ])

        actual = errors.categorical_crossentropy(predicted, target)
        expected = 4.68

        self.assertAlmostEqual(actual.eval(), expected, places=2)
