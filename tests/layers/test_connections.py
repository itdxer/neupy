import numpy as np

from neupy import layers, algorithms
from neupy.utils import asfloat
from neupy.layers import *

from base import BaseTestCase


class ConnectionsTestCase(BaseTestCase):
    def test_connection_initializations(self):
        possible_connections = (
            (2, 3, 1),
            [Input(2), Sigmoid(3), Tanh(1)],
            Input(2) > Relu(10) > Tanh(1),
        )

        for connection in possible_connections:
            network = algorithms.GradientDescent(connection)
            self.assertEqual(len(network.layers), 3)

    def test_connection_inside_connection_mlp(self):
        connection = [
            layers.Input(2),
            layers.Relu(10),
            layers.Relu(4) > layers.Relu(7),
            layers.Relu(3) > layers.Relu(1),
        ]
        expected_sizes = [2, 10, 4, 7, 3, 1]

        network = algorithms.GradientDescent(connection)
        for layer, expected_size in zip(network.layers, expected_sizes):
            self.assertEqual(expected_size, layer.size)

    def test_connection_inside_connection_conv(self):
        connection = [
            layers.Input((1, 28, 28)),

            layers.Convolution((8, 3, 3)) > layers.Relu(),
            layers.Convolution((8, 3, 3)) > layers.Relu(),
            layers.MaxPooling((2, 2)),

            layers.Reshape(),
            layers.Softmax(1),
        ]

        network = algorithms.GradientDescent(connection)
        self.assertEqual(8, len(network.layers))

        self.assertIsInstance(network.layers[1], layers.Convolution)
        self.assertIsInstance(network.layers[2], layers.Relu)
        self.assertIsInstance(network.layers[3], layers.Convolution)
        self.assertIsInstance(network.layers[4], layers.Relu)
        self.assertIsInstance(network.layers[5], layers.MaxPooling)

    def test_connection_shapes(self):
        connection = Input(2) > Relu(10) > Tanh(1)

        self.assertEqual(connection.input_shape, (2,))
        self.assertEqual(connection.output_shape, (1,))

    def test_connection_output(self):
        input_value = asfloat(np.random.random((10, 2)))

        connection = Input(2) > Relu(10) > Relu(1)
        connection.initialize()
        output_value = connection.output(input_value).eval()

        self.assertEqual(output_value.shape, (10, 1))
