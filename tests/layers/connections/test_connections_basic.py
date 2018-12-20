import numpy as np

from neupy import layers, algorithms
from neupy.utils import asfloat

from base import BaseTestCase


class ConnectionsTestCase(BaseTestCase):
    def test_connection_initializations(self):
        possible_connections = (
            layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1),

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
            layers.Input((28, 28, 1)),

            layers.Convolution((3, 3, 8)) > layers.Relu(),
            layers.Convolution((3, 3, 8)) > layers.Relu(),
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

    def test_connection_wrong_number_of_input_values(self):
        input_value_1 = asfloat(np.random.random((10, 2)))
        input_value_2 = asfloat(np.random.random((10, 2)))

        connection = layers.Input(2) > layers.Relu(10) > layers.Relu(1)

        with self.assertRaisesRegexp(ValueError, "but 2 inputs was provided"):
            connection.output(input_value_1, input_value_2)

    def test_one_to_many_parallel_connection_output(self):
        input_connection = layers.Input(4)
        parallel_connections = layers.parallel(
            layers.Linear(11),
            layers.Linear(12),
            layers.Linear(13),
        )
        layers.join(input_connection, parallel_connections)

        input_value = asfloat(np.random.random((10, 4)))
        actual_output = self.eval(parallel_connections.output(input_value))

        self.assertEqual(actual_output[0].shape, (10, 11))
        self.assertEqual(actual_output[1].shape, (10, 12))
        self.assertEqual(actual_output[2].shape, (10, 13))

    def test_many_to_many_parallel_connection_output(self):
        connection = layers.parallel(
            layers.Input(1) > layers.Linear(11),
            layers.Input(2) > layers.Linear(12),
            layers.Input(3) > layers.Linear(13),
        )

        input_value_1 = asfloat(np.random.random((10, 1)))
        input_value_2 = asfloat(np.random.random((20, 2)))
        input_value_3 = asfloat(np.random.random((30, 3)))

        actual_output = self.eval(
            connection.output(input_value_1, input_value_2, input_value_3))

        self.assertEqual(actual_output[0].shape, (10, 11))
        self.assertEqual(actual_output[1].shape, (20, 12))
        self.assertEqual(actual_output[2].shape, (30, 13))
