import numpy as np

from neupy import estimators
from neupy.utils import asfloat

from base import BaseTestCase


class ErrorFuncTestCase(BaseTestCase):
    def test_mse(self):
        actual = np.array([0, 1, 2, 3])
        predicted = np.array([3, 2, 1, 0])
        self.assertEqual(5, estimators.mse(actual, predicted))

        actual = np.array([
            [0, 1],
            [2, 3],
            [4, 5],
        ])
        predicted = np.array([
            [5, 4],
            [3, 2],
            [1, 0],
        ])
        self.assertAlmostEqual(
            asfloat(70 / 6.),
            estimators.mse(actual, predicted)
        )

    def test_binary_crossentropy(self):
        predicted = np.array([0.1, 0.9, 0.2, 0.5])
        actual = np.array([0, 1, 0, 1])

        error = estimators.binary_crossentropy(actual, predicted)
        self.assertAlmostEqual(0.28, error, places=2)

    def test_categorical_crossentropy(self):
        predicted = np.array([
            [0.1, 0.9],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.5, 0.5],
        ])
        actual = np.array([
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
        ])

        error = estimators.categorical_crossentropy(actual, predicted)
        self.assertAlmostEqual(0.28, error, places=2)

    def test_mae(self):
        predicted = np.array([1, 2, 3])
        target = np.array([3, 2, 1])

        actual = estimators.mae(target, predicted)
        self.assertAlmostEqual(actual, 4 / 3.)

    def test_rmse(self):
        actual = np.array([0, 1, 2, 3])
        predicted = np.array([3, 2, 1, 0])
        self.assertEqual(
            asfloat(np.sqrt(5)),
            estimators.rmse(actual, predicted)
        )

    def test_msle(self):
        actual = np.e ** (np.array([1, 2, 3, 4])) - 1
        predicted = np.e ** (np.array([4, 3, 2, 1])) - 1
        self.assertEqual(5, estimators.msle(actual, predicted))

    def test_rmsle(self):
        actual = np.e ** (np.array([1, 2, 3, 4])) - 1
        predicted = np.e ** (np.array([4, 3, 2, 1])) - 1
        self.assertEqual(
            asfloat(np.sqrt(5)),
            estimators.rmsle(actual, predicted)
        )
