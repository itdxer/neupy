import theano
import theano.tensor as T
import numpy as np

from neupy import layers, init
from neupy.utils import asfloat
from neupy.exceptions import LayerConnectionError

from base import BaseTestCase


class ElementwiseTestCase(BaseTestCase):
    def test_elementwise_basic(self):
        elem_layer = layers.Elementwise(merge_function=T.add)

        x1 = T.matrix()
        x2 = T.matrix()
        y = theano.function([x1, x2], elem_layer.output(x1, x2))

        x1_matrix = asfloat(np.random.random((10, 2)))
        x2_matrix = asfloat(np.random.random((10, 2)))

        expected_output = x1_matrix + x2_matrix
        actual_output = y(x1_matrix, x2_matrix)
        np.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_elementwise_initialize(self):
        # Suppose not to fail if you initialize
        # it without connection
        elem_layer = layers.Elementwise()
        elem_layer.initialize()

    def test_elementwise_single_input(self):
        elem_layer = layers.Elementwise()
        output = elem_layer.output(None)
        self.assertEqual(output, None)

    def test_elementwise_init_error(self):
        input_layer_1 = layers.Input(10)
        input_layer_2 = layers.Input(20)
        elem_layer = layers.Elementwise()

        layers.join(input_layer_1, elem_layer)
        with self.assertRaises(LayerConnectionError):
            layers.join(input_layer_2, elem_layer)

    def test_elementwise_not_function(self):
        with self.assertRaises(ValueError):
            not_callable_object = (1, 2, 3)
            layers.Elementwise(merge_function=not_callable_object)

    def test_elementwise_output_shape_no_connection(self):
        elem_layer = layers.Elementwise()
        self.assertEqual(elem_layer.output_shape, None)

    def test_elementwise_in_connections(self):
        input_layer = layers.Input(2)
        hidden_layer_1 = layers.Relu(1, weight=init.Constant(1),
                                     bias=init.Constant(0))
        hidden_layer_2 = layers.Relu(1, weight=init.Constant(2),
                                     bias=init.Constant(0))
        elem_layer = layers.Elementwise(merge_function=T.add)

        connection = layers.join(input_layer, hidden_layer_1, elem_layer)
        connection = layers.join(input_layer, hidden_layer_2, elem_layer)
        connection.initialize()

        self.assertEqual(elem_layer.output_shape, (1,))

        x = T.matrix()
        y = theano.function([x], connection.output(x))

        test_input = asfloat(np.array([
            [0, 1],
            [-1, -1],
        ]))
        actual_output = y(test_input)
        expected_output = np.array([
            [3],
            [0],
        ])
        np.testing.assert_array_almost_equal(expected_output, actual_output)


class ConcatenateTestCase(BaseTestCase):
    def test_concatenate_basic(self):
        concat_layer = layers.Concatenate(axis=1)

        x1 = T.tensor4()
        x2 = T.tensor4()
        y = theano.function([x1, x2], concat_layer.output(x1, x2))

        x1_tensor4 = asfloat(np.random.random((1, 2, 3, 4)))
        x2_tensor4 = asfloat(np.random.random((1, 8, 3, 4)))
        output = y(x1_tensor4, x2_tensor4)

        self.assertEqual((1, 10, 3, 4), output.shape)

    def test_concatenate_init_error(self):
        input_layer_1 = layers.Input((3, 28, 28))
        input_layer_2 = layers.Input((1, 28, 28))
        concat_layer = layers.Concatenate(axis=2)

        layers.join(input_layer_1, concat_layer)
        with self.assertRaises(LayerConnectionError):
            layers.join(input_layer_2, concat_layer)

    def test_concatenate_conv_layers(self):
        input_layer = layers.Input((3, 28, 28))
        hidden_layer_1 = layers.Convolution((7, 5, 5))
        hidden_layer_21 = layers.Convolution((1, 3, 3))
        hidden_layer_22 = layers.Convolution((4, 3, 3))
        concat_layer = layers.Concatenate(axis=1)

        connection = layers.join(input_layer, hidden_layer_1, concat_layer)
        connection = layers.join(input_layer, hidden_layer_21,
                                 hidden_layer_22, concat_layer)
        connection.initialize()

        self.assertEqual((11, 24, 24), concat_layer.output_shape)

        x = T.tensor4()
        y = theano.function([x], connection.output(x))

        x_tensor4 = asfloat(np.random.random((5, 3, 28, 28)))

        actual_output = y(x_tensor4)
        self.assertEqual((5, 11, 24, 24), actual_output.shape)

    def test_elementwise_concatenate(self):
        # Suppose not to fail if you initialize
        # it without connection
        layer = layers.Concatenate()
        layer.initialize()
