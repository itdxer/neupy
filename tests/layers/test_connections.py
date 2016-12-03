import unittest

import numpy as np
import theano
import theano.tensor as T

from neupy import layers, algorithms
from neupy.utils import asfloat, as_tuple
from neupy.layers.connections import is_sequential
from neupy.layers.connections.graph import (merge_dicts_with_list,
                                            does_layer_expect_one_input)

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

        for connection in possible_connections:
            network = algorithms.GradientDescent(connection)
            self.assertEqual(len(network.layers), 3, msg=connection)

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
        output_value = connection.output(input_value).eval()

        self.assertEqual(output_value.shape, (10, 1))


class ConnectionTypesTestCase(BaseTestCase):
    def test_inline_connections(self):
        input_layer = layers.Input(784)
        conn = input_layer > layers.Sigmoid(20)
        conn = conn > layers.Sigmoid(10)

        self.assertEqual(3, len(conn))

        in_sizes = [784, 784, 20]
        out_sizes = [784, 20, 10]

        for layer, in_size, out_size in zip(conn, in_sizes, out_sizes):
            self.assertEqual(layer.input_shape, as_tuple(in_size),
                             msg="Layer: {}".format(layer))
            self.assertEqual(layer.output_shape, as_tuple(out_size),
                             msg="Layer: {}".format(layer))

    def test_connection_shape_multiple_inputs(self):
        input_layer_1 = layers.Input(10)
        input_layer_2 = layers.Input(20)
        conn = [input_layer_1, input_layer_2] > layers.Concatenate()

        self.assertEqual(conn.input_shape, [(10,), (20,)])

    def test_connection_shape_multiple_outputs(self):
        conn = layers.Input(10) > [layers.Sigmoid(1), layers.Sigmoid(2)]

        self.assertEqual(conn.output_shape, [(1,), (2,)])

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

        self.assertEqual(conn1.output_shape, as_tuple(60))
        self.assertEqual(conn2.output_shape, as_tuple(30))
        self.assertEqual(conn3.output_shape, as_tuple(40))

    def test_save_link_to_assigned_connections(self):
        # Tree structure:
        #
        #                       Sigmoid(10)
        #                      /
        # Input(10) - Sigmoid(5)
        #                      \
        #                       Softmax(10)
        #
        input_layer = layers.Input(10)
        minimized = input_layer > layers.Sigmoid(5)
        reconstructed = minimized > layers.Sigmoid(10)
        classifier = minimized > layers.Softmax(20)

        x = T.matrix()
        y_minimized = theano.function([x], minimized.output(x))
        y_reconstructed = theano.function([x], reconstructed.output(x))
        y_classifier = theano.function([x], classifier.output(x))

        x_matrix = asfloat(np.random.random((3, 10)))
        minimized_output = y_minimized(x_matrix)
        self.assertEqual((3, 5), minimized_output.shape)

        reconstructed_output = y_reconstructed(x_matrix)
        self.assertEqual((3, 10), reconstructed_output.shape)

        classifier_output = y_classifier(x_matrix)
        self.assertEqual((3, 20), classifier_output.shape)

    def test_dict_based_inputs_into_connection(self):
        # Tree structure:
        #
        # Input(10) - Sigmoid(5) - Sigmoid(10)
        #
        input_layer = layers.Input(10)
        hidden_layer = layers.Sigmoid(5)
        output_layer = layers.Sigmoid(10)

        minimized = input_layer > hidden_layer
        reconstructed = minimized > output_layer

        x = T.matrix()
        y_minimized = theano.function([x], minimized.output(x))

        x_matrix = asfloat(np.random.random((3, 10)))
        minimized_output = y_minimized(x_matrix)
        self.assertEqual((3, 5), minimized_output.shape)

        h_output = T.matrix()
        y_reconstructed = theano.function(
            [h_output],
            reconstructed.output({output_layer: h_output})
        )
        reconstructed_output = y_reconstructed(minimized_output)
        self.assertEqual((3, 10), reconstructed_output.shape)

    def test_connections_with_complex_parallel_relations(self):
        input_layer = layers.Input((3, 5, 5))
        connection = layers.join(
            [[
                layers.Convolution((8, 1, 1)),
            ], [
                layers.Convolution((4, 1, 1)),
                [[
                    layers.Convolution((2, 1, 3), padding=(0, 1)),
                ], [
                    layers.Convolution((2, 3, 1), padding=(1, 0)),
                ]],
            ], [
                layers.Convolution((8, 1, 1)),
                layers.Convolution((4, 3, 3), padding=1),
                [[
                    layers.Convolution((2, 1, 3), padding=(0, 1)),
                ], [
                    layers.Convolution((2, 3, 1), padding=(1, 0)),
                ]],
            ], [
                layers.MaxPooling((3, 3), stride=(1, 1), padding=1),
                layers.Convolution((8, 1, 1)),
            ]],
            layers.Concatenate(),
        )

        self.assertEqual(connection.input_shape, [None, None, None, None])

        # Connect them at the end, because we need to make
        # sure tha parallel connections defined without
        # input shapes
        connection = input_layer > connection
        self.assertEqual((24, 5, 5), connection.output_shape)


class InlineConnectionsTestCase(BaseTestCase):
    def test_inline_connection_with_parallel_connection(self):
        left_branch = layers.join(
            layers.Convolution((32, 3, 3)),
            layers.Relu(),
            layers.MaxPooling((2, 2)),
        )

        right_branch = layers.join(
            layers.Convolution((16, 7, 7)),
            layers.Relu(),
        )

        input_layer = layers.Input((3, 10, 10))
        concat = layers.Concatenate()

        network_concat = input_layer > [left_branch, right_branch] > concat
        network = network_concat > layers.Reshape() > layers.Softmax()

        self.assertEqual(network_concat.input_shape, (3, 10, 10))
        self.assertEqual(network_concat.output_shape, (48, 4, 4))

        self.assertEqual(network.input_shape, (3, 10, 10))
        self.assertEqual(network.output_shape, (768,))

    def test_inline_connection_wtih_different_pointers(self):
        relu_2 = layers.Relu(2)

        connection_1 = layers.Input(1) > relu_2 > layers.Relu(3)
        connection_2 = relu_2 > layers.Relu(4)
        connection_3 = layers.Input(1) > relu_2

        self.assertEqual(connection_1.input_shape, (1,))
        self.assertEqual(connection_1.output_shape, (3,))

        self.assertEqual(connection_2.input_shape, (1,))
        self.assertEqual(connection_2.output_shape, (4,))

        self.assertEqual(connection_3.input_shape, (1,))
        self.assertEqual(connection_3.output_shape, (2,))

        self.assertIn(relu_2, connection_2.input_layers)
        self.assertIn(relu_2, connection_3.output_layers)

    @unittest.skip("Bug has't been fixed yet")
    def test_inline_connection_order(self):
        input_layer_1 = layers.Input(1)
        relu_2 = layers.Relu(2)
        relu_3 = layers.Relu(3)

        connection_1 = input_layer_1 > relu_2 > relu_3
        self.assertEqual(list(connection_1), [input_layer_1, relu_2, relu_3])

        input_layer_4 = layers.Input(4)
        relu_5 = layers.Relu(5)
        relu_6 = layers.Relu(6)

        connection_2 = input_layer_4 > relu_5
        self.assertEqual(list(connection_2), [input_layer_4, relu_5])

        connection_3 = relu_5 > relu_6
        self.assertEqual(list(connection_3), [relu_5, relu_6])

    def test_right_shift_inplace_inline_operator(self):
        network = layers.Input(1)
        network >>= layers.Relu(2)
        network >>= layers.Relu(3)

        expected_shapes = [1, 2, 3]
        for layer, expected_shape in zip(network, expected_shapes):
            self.assertEqual(layer.output_shape[0], expected_shape)

    def test_left_shift_inplace_inline_operator(self):
        network = layers.Relu(3)
        network <<= layers.Relu(2)
        network <<= layers.Input(1)

        expected_shapes = [1, 2, 3]
        for layer, expected_shape in zip(network, expected_shapes):
            self.assertEqual(layer.output_shape[0], expected_shape)


class ConnectionSecondaryFunctionsTestCase(BaseTestCase):
    def test_is_sequential_connection(self):
        connection1 = layers.join(
            layers.Input(10),
            layers.Sigmoid(5),
            layers.Sigmoid(1),
        )
        self.assertTrue(is_sequential(connection1))

        layer = layers.Input(10)
        self.assertTrue(is_sequential(layer))

    def test_is_sequential_partial_connection(self):
        connection_2 = layers.Input(10) > layers.Sigmoid(5)
        connection_31 = connection_2 > layers.Sigmoid(1)
        connection_32 = connection_2 > layers.Sigmoid(2)

        concatenate = layers.Concatenate()

        connection_4 = connection_31 > concatenate
        connection_4 = connection_32 > concatenate

        self.assertFalse(is_sequential(connection_4))
        self.assertTrue(is_sequential(connection_31))
        self.assertTrue(is_sequential(connection_32))

    def test_dict_merging(self):
        first_dict = dict(a=[1, 2, 3])
        second_dict = dict(a=[3, 4])

        merged_dict = merge_dicts_with_list(first_dict, second_dict)

        self.assertEqual(['a'], list(merged_dict.keys()))
        self.assertEqual([1, 2, 3, 4], merged_dict['a'])

    def test_does_layer_expect_one_input_function(self):
        with self.assertRaises(ValueError):
            does_layer_expect_one_input('not a layer')

        with self.assertRaisesRegexp(ValueError, 'not a method'):
            class A(object):
                output = 'attribute'

            does_layer_expect_one_input(A)

    def test_partial_connection(self):
        network = layers.Sigmoid(1) > layers.Sigmoid(2)

        layer_1, layer_2 = list(network)

        self.assertEqual(layer_1.input_shape, None)
        self.assertEqual(layer_1.output_shape, (1,))

        self.assertEqual(layer_2.input_shape, (1,))
        self.assertEqual(layer_2.output_shape, (2,))


class TestParallelConnectionsTestCase(BaseTestCase):
    def test_parallel_layer(self):
        input_layer = layers.Input((3, 8, 8))
        parallel_layer = layers.join(
            [[
                layers.Convolution((11, 5, 5)),
            ], [
                layers.Convolution((10, 3, 3)),
                layers.Convolution((5, 3, 3)),
            ]],
            layers.Concatenate(),
        )
        output_layer = layers.MaxPooling((2, 2))

        conn = layers.join(input_layer, parallel_layer)
        output_connection = layers.join(conn, output_layer)

        x = T.tensor4()
        y = theano.function([x], conn.output(x))

        x_tensor4 = asfloat(np.random.random((10, 3, 8, 8)))
        output = y(x_tensor4)
        self.assertEqual(output.shape, (10, 11 + 5, 4, 4))

        output_function = theano.function([x], output_connection.output(x))
        final_output = output_function(x_tensor4)
        self.assertEqual(final_output.shape, (10, 11 + 5, 2, 2))

    def test_parallel_with_joined_connections(self):
        # Should work without errors
        layers.join(
            [
                layers.Convolution((11, 5, 5)) > layers.Relu(),
                layers.Convolution((10, 3, 3)) > layers.Relu(),
            ],
            layers.Concatenate() > layers.Relu(),
        )

    def test_parallel_layer_with_residual_connections(self):
        connection = layers.join(
            layers.Input((3, 8, 8)),
            [[
                layers.Convolution((7, 1, 1)),
                layers.Relu()
            ], [
                # Residual connection
            ]],
            layers.Concatenate(),
        )
        self.assertEqual(connection.output_shape, (10, 8, 8))

    def test_standalone_parallel_connection(self):
        connection = layers.join([
            [layers.Input(10) > layers.Sigmoid(1)],
            [layers.Input(20) > layers.Sigmoid(2)],
        ])

        self.assertEqual(connection.input_shape, [(10,), (20,)])
        self.assertEqual(connection.output_shape, [(1,), (2,)])

        outputs = connection.output(T.matrix())
        self.assertEqual(len(outputs), 2)

        outputs = connection.output(T.matrix(), T.matrix())
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
            connection.output(T.matrix(), T.matrix())

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
