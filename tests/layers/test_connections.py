import numpy as np

from neupy import layers, algorithms
from neupy.utils import asfloat, as_tuple
from neupy.layers import Input, Relu, Tanh, Sigmoid

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
        connection = layers.join(
            layers.Input(2),
            layers.Relu(10),
            layers.Relu(4) > layers.Relu(7),
            layers.Relu(3) > layers.Relu(1),
        )
        expected_sizes = [2, 10, 4, 7, 3, 1]
        for layer, expected_size in zip(connection.layers, expected_sizes):
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

        self.assertIsInstance(connection.layers[1], layers.Convolution)
        self.assertIsInstance(connection.layers[2], layers.Relu)
        self.assertIsInstance(connection.layers[3], layers.Convolution)
        self.assertIsInstance(connection.layers[4], layers.Relu)
        self.assertIsInstance(connection.layers[5], layers.MaxPooling)

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

    def test_inline_connections(self):
        conn = layers.Input(784)
        conn = conn > layers.Sigmoid(20)
        conn = conn > layers.Sigmoid(10)

        self.assertEqual(3, len(conn))
        in_sizes = [784, 784, 20]
        out_sizes = [784, 20, 10]
        for layer, in_size, out_size in zip(conn, in_sizes, out_sizes):
            self.assertEqual(layer.input_shape, as_tuple(in_size))
            self.assertEqual(layer.output_shape, as_tuple(out_size))

    def test_tree_connection_structure(self):
        l0 = layers.Input(1)
        l1 = layers.Sigmoid(10)
        l2 = layers.Sigmoid(20)
        l3 = layers.Sigmoid(30)
        l4 = layers.Sigmoid(40)
        l5 = layers.Sigmoid(50)
        l6 = layers.Sigmoid(60)

        # Tree Structure:
        #
        # l0 - l1 - l5 - l6
        #        \
        #         l2 - l4
        #           \
        #            -- l3
        conn1 = layers.join(l0, l1, l5, l6)
        conn2 = layers.join(l0, l1, l2, l3)
        conn3 = layers.join(l0, l1, l2, l4)

        self.assertEqual(4, len(conn1))
        self.assertEqual(4, len(conn2))
        self.assertEqual(4, len(conn3))

        self.assertEqual(conn1.output_shape, as_tuple(60))
        self.assertEqual(conn2.output_shape, as_tuple(30))
        self.assertEqual(conn3.output_shape, as_tuple(40))
