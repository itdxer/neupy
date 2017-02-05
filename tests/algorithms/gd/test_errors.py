import numpy as np
import theano.tensor as T

from neupy import estimators
from neupy.utils import asfloat
from neupy.algorithms.gd import errors

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
            estimators.mse(actual, predicted),
            places=3
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
        self.assertAlmostEqual(actual, 4 / 3., places=3)

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

    def test_binary_hinge(self):
        targets = np.array([
            [-1, 1, 1],
            [-1, -1, 1],
        ])
        predictions = np.array([
            [-0.1, 0.9, 0.5],
            [0.5, -0.5, 1],
        ])
        expected = np.array([
            [0.9, 0.1, 0.5],
            [1.5, 0.5, 0],
        ]).mean()

        actual = estimators.binary_hinge(targets, predictions)
        self.assertAlmostEqual(expected, actual, places=3)

    def test_categorical_hinge(self):
        targets = np.array([
            [0, 0, 1],
            [1, 0, 0],
        ])
        predictions = np.array([
            [0.1, 0.2, 0.7],
            [0.0, 0.9, 0.1],
        ])
        expected = np.array([0.5, 1.9]).mean()

        actual = estimators.categorical_hinge(targets, predictions)
        self.assertAlmostEqual(expected, actual, places=3)

    def test_categorical_hinge_without_one_hot_encoding(self):
        targets = asfloat(np.array([2, 0]))
        predictions = asfloat(np.array([
            [0.1, 0.2, 0.7],
            [0.0, 0.9, 0.1],
        ]))
        expected = asfloat(np.array([0.5, 1.9]).mean())

        prediction_var = T.matrix()
        target_var = T.vector()

        error_output = errors.categorical_hinge(target_var, prediction_var)
        actual = error_output.eval({prediction_var: predictions,
                                    target_var: targets})
        self.assertAlmostEqual(expected, actual, places=3)

    def test_categorical_hinge_invalid_dimension(self):
        with self.assertRaises(TypeError):
            errors.categorical_hinge(T.tensor3(), T.matrix())

    def test_smallest_positive_number(self):
        epsilon = errors.smallest_positive_number()
        self.assertNotEqual(0, asfloat(1) - (asfloat(1) - asfloat(epsilon)))
        self.assertEqual(0, asfloat(1) - (asfloat(1) - asfloat(epsilon / 10)))
