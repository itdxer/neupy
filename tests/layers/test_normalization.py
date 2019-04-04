import tempfile

import numpy as np
import tensorflow as tf
from scipy import stats

from neupy import layers, algorithms, storage
from neupy.utils import asfloat
from neupy.exceptions import (
    LayerConnectionError,
    WeightInitializationError,
)

from base import BaseTestCase
from helpers import simple_classification


class BatchNormTestCase(BaseTestCase):
    def test_batch_norm_as_shared_variable(self):
        gamma = tf.Variable(
            asfloat(np.ones((1, 2))),
            name='gamma',
            dtype=tf.float32,
        )
        beta = tf.Variable(
            asfloat(2 * np.ones((1, 2))),
            name='beta',
            dtype=tf.float32,
        )

        batch_norm = layers.BatchNorm(gamma=gamma, beta=beta)
        network = layers.join(layers.Input(2), batch_norm)
        network.outputs

        self.assertIs(gamma, batch_norm.gamma)
        self.assertIs(beta, batch_norm.beta)

    def test_simple_batch_norm(self):
        network = layers.Input(10) > layers.BatchNorm()

        input_value = tf.Variable(
            asfloat(np.random.random((30, 10))),
            name='input_value',
            dtype=tf.float32,
        )
        output_value = self.eval(network.output(input_value, training=True))

        self.assertTrue(stats.mstats.normaltest(output_value))
        self.assertAlmostEqual(output_value.mean(), 0, places=3)
        self.assertAlmostEqual(output_value.std(), 1, places=3)

    def test_batch_norm_gamma_beta_params(self):
        default_beta = -3.14
        default_gamma = 4.3
        network = layers.join(
            layers.Input(10),
            layers.BatchNorm(gamma=default_gamma, beta=default_beta)
        )

        input_value = tf.Variable(
            asfloat(np.random.random((30, 10))),
            name='input_value',
            dtype=tf.float32,
        )
        output_value = self.eval(network.output(input_value, training=True))

        self.assertAlmostEqual(output_value.mean(), default_beta, places=3)
        self.assertAlmostEqual(output_value.std(), default_gamma, places=3)

    def test_batch_norm_between_layers(self):
        network = layers.join(
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
        outpu_value = self.eval(network.output(input_value, training=True))
        self.assertEqual(outpu_value.shape, (30, 1))

    def test_batch_norm_in_non_training_state(self):
        network = layers.join(
            layers.Input(10),
            layers.BatchNorm(),
        )
        input_value = tf.Variable(
            asfloat(np.random.random((30, 10))),
            name='input_value',
            dtype=tf.float32,
        )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.assertEqual(len(update_ops), 0)

        output_value = network.output(input_value)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.assertEqual(len(update_ops), 0)

        network.output(input_value, training=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.assertEqual(len(update_ops), 2)

        # Without training your running mean and std suppose to be
        # equal to 0 and 1 respectavely.
        output_value = self.eval(output_value)
        np.testing.assert_array_almost_equal(
            self.eval(input_value),
            output_value,
            decimal=4)

    def test_batch_norm_storage(self):
        x_train, x_test, y_train, y_test = simple_classification()

        batch_norm = layers.BatchNorm()
        gdnet = algorithms.GradientDescent(
            [
                layers.Input(10),
                layers.Relu(5),
                batch_norm,
                layers.Sigmoid(1),
            ],
            batch_size=10,
            verbose=True,  # keep it as `True`
        )
        gdnet.train(x_train, y_train, epochs=5)

        error_before_save = gdnet.score(x_test, y_test)
        mean_before_save = self.eval(batch_norm.running_mean)
        variance_before_save = self.eval(batch_norm.running_inv_std)

        with tempfile.NamedTemporaryFile() as temp:
            storage.save(gdnet, temp.name)
            storage.load(gdnet, temp.name)

            error_after_load = gdnet.score(x_test, y_test)
            mean_after_load = self.eval(batch_norm.running_mean)
            variance_after_load = self.eval(batch_norm.running_inv_std)

            self.assertAlmostEqual(error_before_save, error_after_load)
            np.testing.assert_array_almost_equal(
                mean_before_save, mean_after_load)

            np.testing.assert_array_almost_equal(
                variance_before_save,
                variance_after_load)

    def test_batchnorm_wrong_axes(self):
        message = "Specified axes have to contain only unique values"
        with self.assertRaisesRegexp(ValueError, message):
            layers.BatchNorm(axes=(0, 1, 1))

    def test_batchnorm_wrong_axes_values(self):
        network = layers.join(
            layers.Relu(),
            layers.BatchNorm(),
        )
        message = (
            "Cannot initialize variables for the batch normalization "
            "layer, because input shape is undefined"
        )
        with self.assertRaisesRegexp(WeightInitializationError, message):
            network.create_variables()

    def test_batchnorm_unsuitable_axes_values(self):
        network = layers.join(
            layers.Input((10, 3)),
            layers.BatchNorm(axes=(0, 2, 3)),
        )
        message = (
            "Batch normalization cannot be applied over one of "
            "the axis, because input has only 3 dimensions"
        )
        with self.assertRaisesRegexp(LayerConnectionError, message):
            network.create_variables()

    def test_batchnorm_unknown_dimension(self):
        network = layers.join(
            layers.Input((10, 10, None)),
            layers.BatchNorm(),
        )
        message = (
            "Cannot create variables for batch normalization, because "
            "input has unknown dimension #3 \(0-based indices\). "
            "Input shape: \(\?, 10, 10, \?\)"
        )
        with self.assertRaisesRegexp(WeightInitializationError, message):
            network.create_variables()


class LocalResponseNormTestCase(BaseTestCase):
    def test_local_response_norm_exceptions(self):
        with self.assertRaisesRegexp(ValueError, "Only works with odd"):
            layers.LocalResponseNorm(depth_radius=2)

        with self.assertRaises(LayerConnectionError):
            layers.join(layers.Input(10), layers.LocalResponseNorm())

    def test_local_response_normalization_layer(self):
        network = layers.join(
            layers.Input((1, 1, 1)),
            layers.LocalResponseNorm(),
        )

        x_tensor = asfloat(np.ones((1, 1, 1, 1)))
        actual_output = self.eval(network.output(x_tensor, training=True))
        expected_output = np.array([0.59458]).reshape((-1, 1, 1, 1))

        np.testing.assert_array_almost_equal(
            expected_output, actual_output, decimal=5)


class GroupNormTestCase(BaseTestCase):
    def test_group_norm_repr(self):
        layer = layers.GroupNorm(4)
        self.assertEqual(
            str(layer),
            (
                "GroupNorm(n_groups=4, beta=Constant(0), "
                "gamma=Constant(1), epsilon=1e-05, name='group-norm-1')"
            )
        )

    def test_group_norm_connection_exception(self):
        message = "Cannot divide 11 input channels into 4 groups"
        with self.assertRaisesRegexp(LayerConnectionError, message):
            layers.join(
                layers.Input((10, 10, 11)),
                layers.GroupNorm(4),
            )

        message = (
            "Group normalization layer expects 4 "
            "dimensional input, got 3 instead."
        )
        with self.assertRaisesRegexp(LayerConnectionError, message):
            layers.join(
                layers.Input((10, 12)),
                layers.GroupNorm(4),
            )

    def test_group_norm_weight_init_exception(self):
        network = layers.join(
            layers.Input((10, 10, None)),
            layers.GroupNorm(4),
        )

        message = (
            "Cannot initialize variables when number of "
            "channels is unknown."
        )
        with self.assertRaisesRegexp(WeightInitializationError, message):
            network.create_variables()

        with self.assertRaisesRegexp(WeightInitializationError, message):
            network.outputs

    def test_group_norm(self):
        network = layers.join(
            layers.Input((10, 10, 12)),
            layers.GroupNorm(4),
        )
        self.assertShapesEqual(network.input_shape, (None, 10, 10, 12))
        self.assertShapesEqual(network.output_shape, (None, 10, 10, 12))

        input = np.random.random((7, 10, 10, 12))
        actual_output = self.eval(network.output(input))
        self.assertEqual(actual_output.shape, (7, 10, 10, 12))

    def test_group_norm_unknown_shape(self):
        network = layers.join(
            layers.Input((None, None, 16)),
            layers.GroupNorm(4),
        )

        input = np.random.random((7, 6, 6, 16))
        actual_output = self.eval(network.output(input))
        self.assertEqual(actual_output.shape, (7, 6, 6, 16))
