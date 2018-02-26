import numpy as np

from neupy import layers, algorithms
from neupy.utils import asfloat

from base import BaseTestCase


class ConnectionsTestCase(BaseTestCase):
    def test_connection_initializations(self):
        possible_connections = (
            (2, 3, 1),

            # as a list
            [layers.Input(2), layers.Sigmoid(3), layers.Tanh(1)],

            # as forward sequence with inline operators
            layers.Input(2) > layers.Relu(10) > layers.Tanh(1),
            layers.Input(2) >> layers.Relu(10) >> layers.Tanh(1),

            # as backward sequence with inline operators
            layers.Tanh(1) < layers.Relu(10) < layers.Input(2),
            layers.Tanh(1) << layers.Relu(10) << layers.Input(2),
        )

        for i, connection in enumerate(possible_connections, start=1):
            network = algorithms.GradientDescent(connection)
            message = "[Test #{}] Connection: {}".format(i, connection)
            self.assertEqual(len(network.layers), 3, msg=message)

    def test_connection_inside_connection_mlp(self):
        connection = layers.join(
            layers.Input(2),
            layers.Relu(10),
            layers.Relu(4) > layers.Relu(7),
            layers.Relu(3) > layers.Relu(1),
        )

        expected_sizes = [2, 10, 4, 7, 3, 1]
        for layer, expected_size in zip(connection, expected_sizes):
            self.assertEqual(expected_size, layer.size)

    def test_connection_inside_connection_conv(self):
        connection = layers.join(
            layers.Input((1, 28, 28)),

            layers.Convolution((8, 3, 3)) > layers.Relu(),
            layers.Convolution((8, 3, 3)) > layers.Relu(),
            layers.MaxPooling((2, 2)),

            layers.Reshape(),
            layers.Softmax(1),
        )

        self.assertEqual(8, len(connection))

        expected_order = [
            layers.Input, layers.Convolution, layers.Relu,
            layers.Convolution, layers.Relu, layers.MaxPooling,
            layers.Reshape, layers.Softmax
        ]
        for actual_layer, expected_layer in zip(connection, expected_order):
            self.assertIsInstance(actual_layer, expected_layer)

    def test_connection_shapes(self):
        connection = layers.Input(2) > layers.Relu(10) > layers.Tanh(1)

        self.assertEqual(connection.input_shape, (2,))
        self.assertEqual(connection.output_shape, (1,))

    def test_connection_output(self):
        input_value = asfloat(np.random.random((10, 2)))

        connection = layers.Input(2) > layers.Relu(10) > layers.Relu(1)
        output_value = self.eval(connection.output(input_value))

        self.assertEqual(output_value.shape, (10, 1))
