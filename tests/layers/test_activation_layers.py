import math

import numpy as np
import theano
import theano.tensor as T

from neupy.utils import asfloat
from neupy import layers, algorithms, init

from base import BaseTestCase
from data import simple_classification


class ActivationLayersTestCase(BaseTestCase):
    def test_activation_layers_without_size(self):
        input_data = np.array([1, 2, -1, 10])
        expected_output = np.array([1, 2, 0, 10])

        layer = layers.Relu()
        actual_output = layer.output(input_data)

        np.testing.assert_array_equal(actual_output, expected_output)

    def test_sigmoid_layer(self):
        layer1 = layers.Sigmoid(1)
        self.assertGreater(1, layer1.activation_function(1).eval())

    def test_sigmoid_repr_without_size(self):
        layer = layers.Sigmoid()
        self.assertEqual("Sigmoid()", str(layer))

    def test_hard_sigmoid_layer(self):
        layer1 = layers.HardSigmoid(6)

        test_value = asfloat(np.array([[-3, -2, -1, 0, 1, 2]]))
        expected = np.array([[0, 0.1, 0.3, 0.5, 0.7, 0.9]])

        x = T.matrix()
        output = layer1.activation_function(x).eval({x: test_value})

        np.testing.assert_array_almost_equal(output, expected)

    def test_step_layer(self):
        layer1 = layers.Step(1)

        input_vector = theano.shared(np.array([-10, -1, 0, 1, 10]))
        expected = np.array([0, 0, 0, 1, 1])
        output = layer1.activation_function(input_vector).eval()
        np.testing.assert_array_equal(output, expected)

    def test_linear_layer(self):
        layer = layers.Linear(1)
        self.assertEqual(layer.activation_function(1), 1)

    def test_tanh_layer(self):
        layer1 = layers.Tanh(1)
        self.assertGreater(1, layer1.activation_function(1).eval())

    def test_relu_layer(self):
        layer = layers.Relu(1)
        self.assertEqual(0, layer.activation_function(-10))
        self.assertEqual(0, layer.activation_function(0))
        self.assertEqual(10, layer.activation_function(10))

        # Test alpha parameter
        input_data = np.array([[10, 1, 0.1, 0, -0.1, -1]]).T
        expected_output = np.array([[10, 1, 0.1, 0, -0.01, -0.1]]).T
        layer = layers.Relu(1, alpha=0.1)

        actual_output = layer.activation_function(input_data)

        if not isinstance(actual_output, np.ndarray):
            # For Theano >= 0.9.0
            actual_output = actual_output.eval()

        np.testing.assert_array_almost_equal(
            expected_output,
            actual_output
        )

    def test_leaky_relu(self):
        input_data = np.array([[10, 1, 0.1, 0, -0.1, -1]]).T
        expected_output = np.array([[10, 1, 0.1, 0, -0.001, -0.01]]).T
        layer = layers.LeakyRelu(1)

        actual_output = layer.activation_function(input_data)

        if not isinstance(actual_output, np.ndarray):
            # For Theano >= 0.9.0
            actual_output = actual_output.eval()

        np.testing.assert_array_almost_equal(
            expected_output,
            actual_output
        )

    def test_softplus_layer(self):
        layer = layers.Softplus(1)
        self.assertAlmostEqual(
            math.log(2),
            layer.activation_function(0).eval()
        )

    def test_softmax_layer(self):
        test_input = np.array([[0.5, 0.5, 0.1]])

        softmax_layer = layers.Softmax(3)
        correct_result = np.array([[0.37448695, 0.37448695, 0.25102611]])
        np.testing.assert_array_almost_equal(
            correct_result,
            softmax_layer.activation_function(test_input).eval()
        )

    def test_elu_layer(self):
        test_input = np.array([[10, 1, 0.1, 0, -1]]).T
        expected_output = np.array([
            [10, 1, 0.1, 0, 0.1 * math.exp(-1) - 0.1]
        ]).T

        layer = layers.Elu(alpha=0.1)
        actual_output = layer.activation_function(test_input).eval()

        np.testing.assert_array_almost_equal(
            expected_output,
            actual_output
        )

    def test_linear_layer_withut_bias(self):
        input_layer = layers.Input(10)
        output_layer = layers.Linear(2, weight=init.Constant(0.1), bias=None)
        connection = input_layer > output_layer

        self.assertEqual(output_layer.bias_shape, None)

        input_value = asfloat(np.ones((1, 10)))
        actual_output = connection.output(input_value).eval()
        expected_output = np.ones((1, 2))

        np.testing.assert_array_almost_equal(expected_output, actual_output)

        with self.assertRaises(TypeError):
            layers.Linear(2, weight=None)


class PReluTestCase(BaseTestCase):
    def test_invalid_alpha_axes_parameter(self):
        prelu_layer = layers.PRelu(10, alpha_axes=2)
        with self.assertRaises(ValueError):
            # cannot specify 2-axis, because we only
            # have 0 and 1 axes (2D input)
            layers.Input(10) > prelu_layer

        with self.assertRaises(ValueError):
            # 0-axis is not allowed
            layers.PRelu(10, alpha_axes=0)

    def test_prelu_random_params(self):
        prelu_layer = layers.PRelu(10, alpha=init.XavierNormal())
        layers.Input(10) > prelu_layer

        alpha = prelu_layer.alpha.get_value()
        self.assertEqual(10, np.unique(alpha).size)

    def test_prelu_layer_param_dense(self):
        prelu_layer = layers.PRelu(10, alpha=0.25)
        layers.Input(10) > prelu_layer

        alpha = prelu_layer.alpha.get_value()

        self.assertEqual(alpha.shape, (10,))
        np.testing.assert_array_almost_equal(alpha, np.ones(10) * 0.25)

    def test_prelu_layer_param_conv(self):
        input_layer = layers.Input((3, 10, 10))
        conv_layer = layers.Convolution((5, 3, 3))
        prelu_layer = layers.PRelu(alpha=0.25, alpha_axes=(1, 3))

        input_layer > conv_layer > prelu_layer

        alpha = prelu_layer.alpha.get_value()
        expected_alpha = np.ones((5, 8)) * 0.25

        self.assertEqual(alpha.shape, (5, 8))
        np.testing.assert_array_almost_equal(alpha, expected_alpha)

    def test_prelu_output_by_dense_input(self):
        prelu_layer = layers.PRelu(1, alpha=0.25)
        layers.Input(1) > prelu_layer

        input_data = np.array([[10, 1, 0.1, 0, -0.1, -1]]).T
        expected_output = np.array([[10, 1, 0.1, 0, -0.025, -0.25]]).T
        actual_output = prelu_layer.activation_function(input_data).eval()

        np.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_prelu_output_by_spatial_input(self):
        input_data = asfloat(np.random.random((1, 3, 10, 10)))

        input_layer = layers.Input((3, 10, 10))
        conv_layer = layers.Convolution((5, 3, 3))
        prelu_layer = layers.PRelu(alpha=0.25, alpha_axes=(1, 3))

        connection = input_layer > conv_layer > prelu_layer

        actual_output = input_data
        for layer in connection:
            actual_output = layer.output(actual_output)

        actual_output = actual_output.eval()
        self.assertEqual(actual_output.shape, (1, 5, 8, 8))

    def test_prelu_param_updates(self):
        x_train, _, y_train, _ = simple_classification()
        prelu_layer1 = layers.PRelu(20, alpha=0.25)
        prelu_layer2 = layers.PRelu(1, alpha=0.25)

        gdnet = algorithms.GradientDescent(
            [
                layers.Input(10),
                prelu_layer1,
                prelu_layer2,
            ]
        )

        prelu1_alpha_before_training = prelu_layer1.alpha.get_value()
        prelu2_alpha_before_training = prelu_layer2.alpha.get_value()

        gdnet.train(x_train, y_train, epochs=10)

        prelu1_alpha_after_training = prelu_layer1.alpha.get_value()
        prelu2_alpha_after_training = prelu_layer2.alpha.get_value()

        self.assertTrue(all(np.not_equal(
            prelu1_alpha_before_training,
            prelu1_alpha_after_training,
        )))
        self.assertTrue(all(np.not_equal(
            prelu2_alpha_before_training,
            prelu2_alpha_after_training,
        )))

    def test_prelu_axes_property_exceptions(self):
        with self.assertRaises(ValueError):
            layers.PRelu(20, alpha_axes=(-2, -1))
