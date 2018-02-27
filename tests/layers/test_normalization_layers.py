import unittest
import tempfile

import numpy as np
import tensorflow as tf
from scipy import stats

from neupy import layers, algorithms, storage
from neupy.utils import asfloat
from neupy.exceptions import LayerConnectionError
from neupy.layers.normalization import find_opposite_axes

from base import BaseTestCase
from data import simple_classification


class BatchNormTestCase(BaseTestCase):
    def test_find_pposite_axis_invalid_cases(self):
        with self.assertRaises(ValueError):
            find_opposite_axes(axes=[5], ndim=1)

        with self.assertRaises(ValueError):
            find_opposite_axes(axes=[0, 1], ndim=1)

    def test_batch_norm_as_shared_variable(self):
        gamma = tf.Variable(
            asfloat(np.ones(2)),
            name='gamma',
            dtype=tf.float32,
        )
        beta = tf.Variable(
            asfloat(2 * np.ones(2)),
            name='beta',
            dtype=tf.float32,
        )

        batch_norm = layers.BatchNorm(gamma=gamma, beta=beta)
        layers.Input(10) > batch_norm

        self.assertIs(gamma, batch_norm.gamma)
        self.assertIs(beta, batch_norm.beta)

    def test_find_pposite_axis_valid_cases(self):
        testcases = (
            dict(input_kwargs={'axes': [0, 1], 'ndim': 4},
                 expected_output=[2, 3]),
            dict(input_kwargs={'axes': [], 'ndim': 4},
                 expected_output=[0, 1, 2, 3]),
            dict(input_kwargs={'axes': [0, 1, 2], 'ndim': 3},
                 expected_output=[]),
        )

        for testcase in testcases:
            actual_output = find_opposite_axes(**testcase['input_kwargs'])
            self.assertEqual(actual_output, testcase['expected_output'],
                             msg="Kwargs: ".format(testcase['input_kwargs']))

    def test_simple_batch_norm(self):
        connection = layers.Input(10) > layers.BatchNorm()

        input_value = tf.Variable(
            asfloat(np.random.random((30, 10))),
            name='input_value',
            dtype=tf.float32,
        )
        output_value = self.eval(connection.output(input_value))

        self.assertTrue(stats.mstats.normaltest(output_value))
        self.assertAlmostEqual(output_value.mean(), 0, places=3)
        self.assertAlmostEqual(output_value.std(), 1, places=3)

    def test_batch_norm_gamma_beta_params(self):
        default_beta = -3.14
        default_gamma = 4.3
        connection = layers.join(
            layers.Input(10),
            layers.BatchNorm(gamma=default_gamma, beta=default_beta)
        )

        input_value = tf.Variable(
            asfloat(np.random.random((30, 10))),
            name='input_value',
            dtype=tf.float32,
        )
        output_value = self.eval(connection.output(input_value))

        self.assertAlmostEqual(output_value.mean(), default_beta, places=3)
        self.assertAlmostEqual(output_value.std(), default_gamma, places=3)

    def test_batch_norm_between_layers(self):
        connection = layers.join(
            layers.Input(10),
            layers.Relu(40),
            layers.BatchNorm(),
            layers.Relu(1),
        )

        input_value = tf.Variable(
            asfloat(np.random.random((30, 10))),
            name='input_value',
            dtype=tf.float32,
        )
        outpu_value = self.eval(connection.output(input_value))

        self.assertEqual(outpu_value.shape, (30, 1))

    def test_batch_norm_exceptions(self):
        with self.assertRaises(ValueError):
            # Axis does not exist
            layers.Input(10) > layers.BatchNorm(axes=2)

        with self.assertRaises(ValueError):
            connection = layers.Relu() > layers.BatchNorm()
            connection.initialize()

    def test_batch_norm_in_non_training_state(self):
        batch_norm = layers.BatchNorm()
        layers.Input(10) > batch_norm

        input_value = tf.Variable(
            asfloat(np.random.random((30, 10))),
            name='input_value',
            dtype=tf.float32,
        )

        self.assertEqual(len(batch_norm.updates), 0)

        batch_norm.output(input_value)
        self.assertEqual(len(batch_norm.updates), 2)

        with batch_norm.disable_training_state():
            # Without training your running mean and std suppose to be
            # equal to 0 and 1 respectavely.
            output_value = self.eval(batch_norm.output(input_value))
            np.testing.assert_array_almost_equal(
                self.eval(input_value), output_value, decimal=4)

    def test_batch_norm_storage(self):
        x_train, x_test, y_train, y_test = simple_classification()

        batch_norm = layers.BatchNorm()
        gdnet = algorithms.MinibatchGradientDescent(
            [
                layers.Input(10),
                layers.Relu(5),
                batch_norm,
                layers.Sigmoid(1),
            ],
            batch_size=10,
            verbose=True,
        )
        gdnet.train(x_train, y_train, epochs=5)

        error_before_save = gdnet.prediction_error(x_test, y_test)
        mean_before_save = self.eval(batch_norm.running_mean)
        variance_before_save = self.eval(batch_norm.running_variance)

        with tempfile.NamedTemporaryFile() as temp:
            storage.save(gdnet, temp.name)
            storage.load(gdnet, temp.name)

            error_after_load = gdnet.prediction_error(x_test, y_test)
            mean_after_load = self.eval(batch_norm.running_mean)
            variance_after_load = self.eval(batch_norm.running_variance)

            self.assertAlmostEqual(error_before_save, error_after_load)
            np.testing.assert_array_almost_equal(
                mean_before_save, mean_after_load)

            np.testing.assert_array_almost_equal(
                variance_before_save, variance_after_load)


class LocalResponseNormTestCase(BaseTestCase):
    def test_local_response_norm_exceptions(self):
        with self.assertRaises(ValueError):
            layers.LocalResponseNorm(n=2)

        with self.assertRaises(LayerConnectionError):
            layers.Input(10) > layers.LocalResponseNorm()

    def test_local_response_normalization_layer(self):
        input_layer = layers.Input((1, 1, 1))
        conn = input_layer > layers.LocalResponseNorm()

        x_tensor = asfloat(np.ones((1, 1, 1, 1)))
        actual_output = self.eval(conn.output(x_tensor))
        expected_output = np.array([0.59458]).reshape((-1, 1, 1, 1))

        np.testing.assert_array_almost_equal(
            expected_output, actual_output, decimal=5
        )
