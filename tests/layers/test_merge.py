import tensorflow as tf
import numpy as np

from neupy import layers
from neupy.utils import asfloat
from neupy.exceptions import LayerConnectionError

from base import BaseTestCase


class ElementwiseTestCase(BaseTestCase):
    def test_elementwise_basic(self):
        elem_layer = layers.Elementwise(merge_function=tf.add)

        x1_matrix = asfloat(np.random.random((10, 2)))
        x2_matrix = asfloat(np.random.random((10, 2)))

        expected_output = x1_matrix + x2_matrix
        actual_output = self.eval(elem_layer.output(x1_matrix, x2_matrix))
        np.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_elementwise_exceptions(self):
        with self.assertRaises(ValueError):
            not_callable_object = (1, 2, 3)
            layers.Elementwise(merge_function=not_callable_object)

        with self.assertRaises(ValueError):
            layers.Elementwise(merge_function='wrong-func-name')

        message = "expected multiple inputs"
        with self.assertRaisesRegexp(LayerConnectionError, message):
            layers.join(layers.Input(5), layers.Elementwise('multiply'))

        inputs = layers.parallel(
            layers.Input(2),
            layers.Input(1),
        )
        error_message = "layer have incompatible shapes"
        with self.assertRaisesRegexp(LayerConnectionError, error_message):
            layers.join(inputs, layers.Elementwise('add'))

    def test_elementwise_in_network(self):
        network = layers.join(
            layers.Input(2),
            layers.parallel(
                layers.Relu(1, weight=1, bias=0),
                layers.Relu(1, weight=2, bias=0),
            ),
            layers.Elementwise('add'),
        )
        self.assertShapesEqual(network.input_shape, (None, 2))
        self.assertShapesEqual(network.output_shape, (None, 1))

        test_input = asfloat(np.array([[0, 1], [-1, -1]]))
        actual_output = self.eval(network.output(test_input))
        expected_output = np.array([[3, 0]]).T
        np.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_elementwise_custom_function(self):
        def weighted_sum(a, b):
            return 0.2 * a + 0.8 * b

        network = layers.join(
            layers.Input(2),
            layers.parallel(
                layers.Relu(1, weight=1, bias=0),
                layers.Relu(1, weight=2, bias=0),
            ),
            layers.Elementwise(weighted_sum),
        )
        self.assertShapesEqual(network.input_shape, (None, 2))
        self.assertShapesEqual(network.output_shape, (None, 1))

        test_input = asfloat(np.array([[0, 1], [-1, -1]]))
        actual_output = self.eval(network.output(test_input))
        expected_output = np.array([[1.8, 0]]).T
        np.testing.assert_array_almost_equal(expected_output, actual_output)


class ConcatenateTestCase(BaseTestCase):
    def test_concatenate_basic(self):
        concat_layer = layers.Concatenate(axis=-1)

        x1_tensor4 = asfloat(np.random.random((1, 3, 4, 2)))
        x2_tensor4 = asfloat(np.random.random((1, 3, 4, 8)))
        output = self.eval(concat_layer.output(x1_tensor4, x2_tensor4))

        self.assertEqual((1, 3, 4, 10), output.shape)

    def test_concatenate_different_dim_number(self):
        inputs = layers.parallel(
            layers.Input((28, 28)),
            layers.Input((28, 28, 1)),
        )

        expected_msg = "different number of dimensions"
        with self.assertRaisesRegexp(LayerConnectionError, expected_msg):
            layers.join(inputs, layers.Concatenate(axis=1))

    def test_concatenate_init_error(self):
        inputs = layers.parallel(
            layers.Input((28, 28, 3)),
            layers.Input((28, 28, 1)),
        )

        expected_message = "don't match over dimension #3"
        with self.assertRaisesRegexp(LayerConnectionError, expected_message):
            layers.join(inputs, layers.Concatenate(axis=2))

    def test_concatenate_conv_layers(self):
        network = layers.join(
            layers.Input((28, 28, 3)),
            layers.parallel(
                layers.Convolution((5, 5, 7)),
                layers.join(
                    layers.Convolution((3, 3, 1)),
                    layers.Convolution((3, 3, 4)),
                ),
            ),
            layers.Concatenate(axis=-1)
        )

        self.assertShapesEqual((None, 24, 24, 11), network.output_shape)

        x_tensor4 = asfloat(np.random.random((5, 28, 28, 3)))
        actual_output = self.eval(network.output(x_tensor4))

        self.assertEqual((5, 24, 24, 11), actual_output.shape)
