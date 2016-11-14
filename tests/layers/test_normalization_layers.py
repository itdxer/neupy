import numpy as np
import theano
import theano.tensor as T
from scipy import stats

from neupy import layers
from neupy.utils import asfloat
from neupy.layers.normalization import find_opposite_axes
from neupy.layers.connections import LayerConnectionError

from base import BaseTestCase


class BatchNormTestCase(BaseTestCase):
    def test_find_pposite_axis_invalid_cases(self):
        with self.assertRaises(ValueError):
            find_opposite_axes(axes=[5], ndim=1)

        with self.assertRaises(ValueError):
            find_opposite_axes(axes=[0, 1], ndim=1)

    def test_batch_norm_as_shared_variable(self):
        gamma = theano.shared(value=asfloat(np.ones(2)))
        beta = theano.shared(value=asfloat(2 * np.ones(2)))

        batch_norm = layers.BatchNorm(gamma=gamma, beta=beta)
        connection = layers.Input(10) > batch_norm
        connection.initialize()

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
        connection.initialize()

        input_value = theano.shared(value=np.random.random((30, 10)))
        output_value = connection.output(input_value).eval()

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
        connection.initialize()

        input_value = theano.shared(value=np.random.random((30, 10)))
        output_value = connection.output(input_value).eval()

        self.assertAlmostEqual(output_value.mean(), default_beta, places=3)
        self.assertAlmostEqual(output_value.std(), default_gamma, places=3)

    def test_batch_norm_between_layers(self):
        connection = layers.join(
            layers.Input(10),
            layers.Relu(40),
            layers.BatchNorm(),
            layers.Relu(1),
        )
        connection.initialize()

        input_value = np.random.random((30, 10))
        outpu_value = connection.output(input_value).eval()

        self.assertEqual(outpu_value.shape, (30, 1))

    def test_batch_norm_exceptions(self):
        with self.assertRaises(ValueError):
            # Axis does not exist
            connection = layers.Input(10) > layers.BatchNorm(axes=2)
            connection.initialize()

        with self.assertRaises(ValueError):
            connection = layers.Relu() > layers.BatchNorm()
            connection.initialize()

    def test_batch_norm_in_non_training_state(self):
        batch_norm = layers.BatchNorm()
        connection = layers.Input(10) > batch_norm
        connection.initialize()

        input_value = theano.shared(value=np.random.random((30, 10)))

        self.assertEqual(len(batch_norm.updates), 0)
        batch_norm.output(input_value)
        self.assertEqual(len(batch_norm.updates), 2)

        with batch_norm.disable_training_state():
            # Without training your running mean and std suppose to be
            # equal to 0 and 1 respectavely.
            output_value = batch_norm.output(input_value).eval()
            np.testing.assert_array_almost_equal(
                input_value.get_value(),
                output_value
            )


class LocalResponseNormTestCase(BaseTestCase):
    def test_local_response_norm_exceptions(self):
        with self.assertRaises(ValueError):
            layers.LocalResponseNorm(n=2)

        conn = layers.Input(10) > layers.LocalResponseNorm()
        with self.assertRaises(LayerConnectionError):
            conn.output(T.matrix())

        conn = layers.LocalResponseNorm()
        with self.assertRaises(LayerConnectionError):
            conn.output(T.tensor4())

    def test_local_response_normalization_layer(self):
        input_layer = layers.Input((1, 1, 1))
        conn = input_layer > layers.LocalResponseNorm()

        x = T.tensor4()
        y = theano.function([x], conn.output(x))

        x_tensor = asfloat(np.ones((1, 1, 1, 1)))
        actual_output = y(x_tensor)
        expected_output = np.array([0.59458]).reshape((-1, 1, 1, 1))

        np.testing.assert_array_almost_equal(
            expected_output, actual_output, decimal=5
        )
