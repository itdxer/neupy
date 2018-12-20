import numpy as np

from neupy.utils import asfloat
from neupy.algorithms.gd import objectives

from base import BaseTestCase


class ErrorFuncTestCase(BaseTestCase):
    def test_mse(self):
        actual = np.array([0, 1, 2, 3])
        predicted = np.array([3, 2, 1, 0])
        self.assertEqual(5, self.eval(objectives.mse(actual, predicted)))

        actual = asfloat(np.array([
            [0, 1],
            [2, 3],
            [4, 5],
        ]))
        predicted = asfloat(np.array([
            [5, 4],
            [3, 2],
            [1, 0],
        ]))
        self.assertAlmostEqual(
            asfloat(70 / 6.),
            self.eval(objectives.mse(actual, predicted)),
            places=3
        )

    def test_binary_crossentropy(self):
        predicted = asfloat(np.array([0.1, 0.9, 0.2, 0.5]))
        actual = asfloat(np.array([0, 1, 0, 1]))

        error = objectives.binary_crossentropy(actual, predicted)
        self.assertAlmostEqual(0.28, self.eval(error), places=2)

    def test_categorical_crossentropy(self):
        predicted = asfloat(np.array([
            [0.1, 0.9],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.5, 0.5],
        ]))
        actual = asfloat(np.array([
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
        ]))

        error = objectives.categorical_crossentropy(actual, predicted)
        self.assertAlmostEqual(0.28, self.eval(error), places=2)

    def test_binary_crossentropy_spatial_data(self):
        pred_values = np.array([[
            [0.3, 0.2, 0.1],
            [0.1, 0.1, 0.1],
        ]])
        pred_values = np.transpose(pred_values, (1, 2, 0))
        pred_values = np.expand_dims(pred_values, axis=0)

        true_values = np.array([[
            [1, 0, 1],
            [0, 0, 0],
        ]])
        true_values = np.transpose(true_values, (1, 2, 0))
        true_values = np.expand_dims(true_values, axis=0)

        # Making sure that input values are proper probabilities
        self.assertTrue(np.all(pred_values.sum(axis=-1) < 1))

        error = objectives.binary_crossentropy(true_values, pred_values)
        expected_error = -(
            np.log(0.3) + np.log(0.1) + 3 * np.log(0.9) + np.log(0.8)) / 6

        self.assertAlmostEqual(expected_error, self.eval(error), places=2)

    def test_categorical_crossentropy_spatial_data(self):
        pred_values = np.array([[
            [0.3, 0.2, 0.1],
            [0.1, 0.1, 0.1],
        ], [
            [0.7, 0.8, 0.9],
            [0.9, 0.9, 0.9],
        ]])
        pred_values = np.transpose(pred_values, (1, 2, 0))
        pred_values = np.expand_dims(pred_values, axis=0)

        true_values = np.array([[
            [1, 0, 1],
            [0, 0, 0],
        ], [
            [0, 1, 0],
            [1, 1, 1],
        ]])
        true_values = np.transpose(true_values, (1, 2, 0))
        true_values = np.expand_dims(true_values, axis=0)

        # Making sure that input values are proper probabilities
        self.assertTrue(np.allclose(pred_values.sum(axis=-1), 1))
        self.assertTrue(np.allclose(true_values.sum(axis=-1), 1))

        error = objectives.categorical_crossentropy(true_values, pred_values)
        expected_error = -(
            np.log(0.3) + np.log(0.1) + 3 * np.log(0.9) + np.log(0.8)) / 6

        self.assertAlmostEqual(expected_error, self.eval(error), places=2)

    def test_mae(self):
        predicted = asfloat(np.array([1, 2, 3]))
        target = asfloat(np.array([3, 2, 1]))

        actual = objectives.mae(target, predicted)
        self.assertAlmostEqual(self.eval(actual), 4 / 3., places=3)

    def test_rmse(self):
        actual = asfloat(np.array([0, 1, 2, 3]))
        predicted = asfloat(np.array([3, 2, 1, 0]))
        self.assertAlmostEqual(
            asfloat(np.sqrt(5)),
            self.eval(objectives.rmse(actual, predicted))
        )

    def test_msle(self):
        actual = np.e ** (np.array([1, 2, 3, 4])) - 1
        predicted = np.e ** (np.array([4, 3, 2, 1])) - 1
        self.assertEqual(5, self.eval(objectives.msle(actual, predicted)))

    def test_rmsle(self):
        actual = np.e ** (np.array([1, 2, 3, 4])) - 1
        predicted = np.e ** (np.array([4, 3, 2, 1])) - 1
        self.assertAlmostEqual(
            asfloat(np.sqrt(5)),
            self.eval(objectives.rmsle(actual, predicted))
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

        actual = objectives.binary_hinge(targets, predictions)
        self.assertAlmostEqual(expected, self.eval(actual), places=3)

    def test_categorical_hinge(self):
        targets = asfloat(np.array([
            [0, 0, 1],
            [1, 0, 0],
        ]))
        predictions = asfloat(np.array([
            [0.1, 0.2, 0.7],
            [0.0, 0.9, 0.1],
        ]))
        expected = np.array([0.5, 1.9]).mean()

        actual = objectives.categorical_hinge(targets, predictions)
        self.assertAlmostEqual(expected, self.eval(actual), places=3)
