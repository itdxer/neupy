import numpy as np

from neupy import layers
from neupy.utils import asfloat

from base import BaseTestCase


class TestParallelConnectionsTestCase(BaseTestCase):
    def test_parallel_layer(self):
        input_layer = layers.Input((8, 8, 3))
        parallel_layer = layers.join(
            [[
                layers.Convolution((5, 5, 11)),
            ], [
                layers.Convolution((3, 3, 10)),
                layers.Convolution((3, 3, 5)),
            ]],
            layers.Concatenate(),
        )
        output_layer = layers.MaxPooling((2, 2))

        conn = layers.join(input_layer, parallel_layer)
        output_connection = layers.join(conn, output_layer)

        x_tensor4 = asfloat(np.random.random((10, 8, 8, 3)))
        output = self.eval(conn.output(x_tensor4))
        self.assertEqual(output.shape, (10, 4, 4, 11 + 5))

        final_output = output_connection.predict(x_tensor4)
        self.assertEqual(final_output.shape, (10, 2, 2, 11 + 5))

    def test_parallel_with_joined_connections(self):
        # Should work without errors
        layers.join(
            [
                layers.Convolution((5, 5, 11)) > layers.Relu(),
                layers.Convolution((3, 3, 10)) > layers.Relu(),
            ],
            layers.Concatenate() > layers.Relu(),
        )

    def test_parallel_layer_with_residual_connections(self):
        connection = layers.join(
            layers.Input((8, 8, 3)),
            [[
                layers.Convolution((1, 1, 7)),
                layers.Relu()
            ], [
                # Residual connection
            ]],
            layers.Concatenate(),
        )
        self.assertEqual(connection.output_shape, (8, 8, 10))

    def test_standalone_parallel_connection(self):
        connection = layers.join([
            [layers.Input(10) > layers.Sigmoid(1)],
            [layers.Input(20) > layers.Sigmoid(2)],
        ])

        self.assertEqual(connection.input_shape, [(10,), (20,)])
        self.assertEqual(connection.output_shape, [(1,), (2,)])

        input_value_10 = asfloat(np.random.random((2, 10)))
        input_value_20 = asfloat(np.random.random((2, 20)))

        outputs = connection.output(input_value_10, input_value_20)
        self.assertEqual(len(outputs), 2)

    def test_parallel_connection_initialize_method(self):
        class CustomLayer(layers.BaseLayer):
            initialized = False

            def initialize(self):
                self.initialized = True

        connections = layers.join([
            [CustomLayer(), CustomLayer(), CustomLayer()],
            [CustomLayer(), CustomLayer(), CustomLayer()],
            [CustomLayer(), CustomLayer(), CustomLayer()],
        ])
        connections.initialize()

        for connection in connections:
            for layer in connection:
                self.assertTrue(layer.initialized, msg=layer.name)

    def test_parallel_connection_disable_training_sate(self):
        connections = layers.join([
            [layers.Input(10) > layers.Sigmoid(1)],
            [layers.Input(20) > layers.Sigmoid(2)],
        ])

        all_layers = []
        for connection in connections:
            all_layers.extend(list(connection))

        # Enabled
        for layer in all_layers:
            self.assertTrue(layer.training_state, msg=layer)

        # Disabled
        with connections.disable_training_state():
            for layer in all_layers:
                self.assertFalse(layer.training_state, msg=layer)

        # Enabled
        for layer in all_layers:
            self.assertTrue(layer.training_state, msg=layer)

    def test_parallel_connection_output_exceptions(self):
        connection = layers.join([
            [layers.Input(10) > layers.Sigmoid(1)],
            [layers.Input(20) > layers.Sigmoid(2)],
            [layers.Input(30) > layers.Sigmoid(3)],
        ])

        with self.assertRaises(ValueError):
            # Received only 2 inputs instead of 3
            connection.output(
                np.random.random((2, 10)),
                np.random.random((2, 20)),
            )

    def test_parallel_many_to_many_connection(self):
        relu_layer_1 = layers.Relu(1)
        sigmoid_layer_1 = layers.Sigmoid(1)

        relu_layer_2 = layers.Relu(2)
        sigmoid_layer_2 = layers.Sigmoid(2)

        connection = layers.join(
            [
                sigmoid_layer_1,
                relu_layer_1,
            ], [
                sigmoid_layer_2,
                relu_layer_2,
            ],
        )

        self.assertEqual(connection.input_shape, [None, None])
        self.assertEqual(connection.output_shape, [(2,), (2,)])

        graph = connection.graph

        for layer in [relu_layer_1, sigmoid_layer_1]:
            n_forward_connections = len(graph.forward_graph[layer])
            n_backward_connections = len(graph.backward_graph[layer])

            self.assertEqual(n_forward_connections, 2)
            self.assertEqual(n_backward_connections, 0)

        for layer in [relu_layer_2, sigmoid_layer_2]:
            n_forward_connections = len(graph.forward_graph[layer])
            n_backward_connections = len(graph.backward_graph[layer])

            self.assertEqual(n_forward_connections, 0)
            self.assertEqual(n_backward_connections, 2)
