import unittest

import numpy as np
import theano
import theano.tensor as T

from neupy import layers, algorithms, init
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

    def test_dict_based_inputs_into_connection_with_layer_names(self):
        encoder = layers.Input(10) > layers.Sigmoid(5, name='sigmoid-1')
        decoder = layers.Sigmoid(2, name='sigmoid-2') > layers.Sigmoid(10)

        network = encoder > decoder

        x = T.matrix()
        predict = theano.function([x], network.output({'sigmoid-2': x}))

        x_test = asfloat(np.ones((7, 5)))
        y_predicted = predict(x_test)

        self.assertEqual(y_predicted.shape, (7, 10))

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

    def test_single_input_for_parallel_layers(self):
        left = layers.Input(10, name='input') > layers.Sigmoid(5)
        right = left > layers.Sigmoid(2)
        network = [left, right] > layers.Concatenate()

        self.assertEqual(len(network.input_layers), 1)

        input_layer = network.input_layers[0]
        self.assertEqual(input_layer.name, 'input')

    def test_single_output_for_parallel_layers(self):
        left = layers.Sigmoid(5, name='output')
        right = layers.Relu(10) > left
        network = layers.Input(10) > [left, right]

        self.assertEqual(len(network.output_layers), 1)

        output_layer = network.output_layers[0]
        self.assertEqual(output_layer.name, 'output')


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


class ConnectionCompilationTestCase(BaseTestCase):
    def test_simple_connection_compilation(self):
        input_matrix = asfloat(np.ones((7, 10)))
        expected_output = np.ones((7, 5))

        network = layers.join(
            layers.Input(10),
            layers.Linear(5, weight=init.Constant(0.1), bias=None)
        )

        # Generated input variables
        predict = network.compile()
        actual_output = predict(input_matrix)
        np.testing.assert_array_almost_equal(actual_output, expected_output)

        # Pre-defined input variables
        input_variable = T.matrix('x')
        predict = network.compile(input_variable)
        actual_output = predict(input_matrix)
        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_compilation_multiple_inputs(self):
        input_matrix = asfloat(np.ones((7, 10)))
        expected_output = np.ones((7, 5))

        network = layers.join(
            [[
                layers.Input(10),
            ], [
                layers.Input(10),
            ]],
            layers.Elementwise(),
            layers.Linear(5, weight=init.Constant(0.1), bias=None)
        )

        # Generated input variables
        predict = network.compile()
        actual_output = predict(input_matrix * 0.7, input_matrix * 0.3)
        np.testing.assert_array_almost_equal(actual_output, expected_output)

        # Pre-defined input variables
        input_variable_1 = T.matrix('x1')
        input_variable_2 = T.matrix('x2')

        predict = network.compile(input_variable_1, input_variable_2)
        actual_output = predict(input_matrix * 0.7, input_matrix * 0.3)
        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_compilation_exceptions(self):
        network = [layers.Input(2), layers.Input(2)] > layers.Concatenate()
        with self.assertRaises(ValueError):
            # n_input_vars != n_input_layers
            network.compile(T.matrix('x'), T.matrix('y'), T.matrix('z'))

    def test_compilation_multiple_outputs(self):
        input_matrix = asfloat(np.ones((7, 10)))
        expected_output_1 = np.ones((7, 5))
        expected_output_2 = np.ones((7, 2))

        network = layers.join(
            layers.Input(10),
            [[
                layers.Linear(5, weight=init.Constant(0.1), bias=None)
            ], [
                layers.Linear(2, weight=init.Constant(0.1), bias=None)
            ]]
        )
        predict = network.compile()

        actual_output_1, actual_output_2 = predict(input_matrix)

        np.testing.assert_array_almost_equal(
            actual_output_1, expected_output_1)

        np.testing.assert_array_almost_equal(
            actual_output_2, expected_output_2)


class SliceLayerConnectionsTestCase(BaseTestCase):
    def test_change_output_layer(self):
        network = layers.join(
            layers.Input(10, name='input-1'),
            layers.Relu(5, name='relu-1'),
            layers.Relu(1, name='relu-2'),
        )

        self.assertEqual(network.input_shape, (10,))
        self.assertEqual(network.output_shape, (1,))
        self.assertEqual(len(network), 3)

        relu_1_network = network.end('relu-1')
        self.assertEqual(relu_1_network.input_shape, (10,))
        self.assertEqual(relu_1_network.output_shape, (5,))
        self.assertEqual(len(relu_1_network.layers), 2)

        predict = relu_1_network.compile()
        x_test = asfloat(np.ones((7, 10)))
        y_predicted = predict(x_test)
        self.assertEqual(y_predicted.shape, (7, 5))

    def test_select_network_branch(self):
        network = layers.join(
            layers.Input(10, name='input-1'),
            [[
                layers.Relu(1, name='relu-1'),
            ], [
                layers.Relu(2, name='relu-2'),
            ]]
        )

        self.assertEqual(network.input_shape, (10,))
        self.assertEqual(network.output_shape, [(1,), (2,)])
        self.assertEqual(len(network), 3)

        relu_1_network = network.end('relu-1')
        self.assertEqual(relu_1_network.input_shape, (10,))
        self.assertEqual(relu_1_network.output_shape, (1,))
        self.assertEqual(len(relu_1_network), 2)

        predict = relu_1_network.compile()
        x_test = asfloat(np.ones((7, 10)))
        y_predicted = predict(x_test)
        self.assertEqual(y_predicted.shape, (7, 1))

        relu_2_network = network.end('relu-2')
        self.assertEqual(relu_2_network.input_shape, (10,))
        self.assertEqual(relu_2_network.output_shape, (2,))
        self.assertEqual(len(relu_2_network), 2)

    def test_cut_output_layers_in_sequence(self):
        network = layers.join(
            layers.Input(10, name='input-1'),
            layers.Relu(5, name='relu-1'),
            layers.Relu(1, name='relu-2'),
        )

        self.assertEqual(network.input_shape, (10,))
        self.assertEqual(network.output_shape, (1,))
        self.assertEqual(len(network), 3)

        cutted_network = network.end('relu-1').end('input-1')
        self.assertEqual(cutted_network.input_shape, (10,))
        self.assertEqual(cutted_network.output_shape, (10,))
        self.assertEqual(len(cutted_network), 1)

        predict = cutted_network.compile()
        x_test = asfloat(np.ones((7, 10)))
        y_predicted = predict(x_test)
        self.assertEqual(y_predicted.shape, (7, 10))

    def test_cut_using_layer_object(self):
        relu = layers.Relu(2)
        network = layers.Input(10) > relu > layers.Sigmoid(1)

        self.assertEqual(network.input_shape, (10,))
        self.assertEqual(network.output_shape, (1,))
        self.assertEqual(len(network), 3)

        cutted_network = network.end(relu)
        self.assertEqual(cutted_network.input_shape, (10,))
        self.assertEqual(cutted_network.output_shape, (2,))
        self.assertEqual(len(cutted_network), 2)

    def test_unknown_layer_name_exception(self):
        network = layers.join(
            layers.Input(10, name='input-1'),
            layers.Relu(5, name='relu-1'),
            layers.Relu(1, name='relu-2'),
        )
        with self.assertRaises(NameError):
            network.end('abc')

    def test_change_input_layer(self):
        network = layers.join(
            layers.Input(10, name='input-1'),
            layers.Relu(5, name='relu-1'),
            layers.Relu(1, name='relu-2'),
        )

        self.assertEqual(network.input_shape, (10,))
        self.assertEqual(network.output_shape, (1,))
        self.assertEqual(len(network), 3)

        relu_1_network = network.start('relu-1')
        self.assertEqual(relu_1_network.input_shape, (10,))
        self.assertEqual(relu_1_network.output_shape, (1,))
        self.assertEqual(len(relu_1_network), 2)

        predict = relu_1_network.compile()
        x_test = asfloat(np.ones((7, 10)))
        y_predicted = predict(x_test)
        self.assertEqual(y_predicted.shape, (7, 1))

    def test_cut_input_and_output_layers(self):
        network = layers.join(
            layers.Input(10, name='input-1'),
            layers.Relu(8, name='relu-0'),
            layers.Relu(5, name='relu-1'),
            layers.Relu(2, name='relu-2'),
            layers.Relu(1, name='relu-3'),
        )

        self.assertEqual(network.input_shape, (10,))
        self.assertEqual(network.output_shape, (1,))
        self.assertEqual(len(network), 5)

        cutted_network = network.start('relu-1').end('relu-2')

        self.assertEqual(cutted_network.input_shape, (8,))
        self.assertEqual(cutted_network.output_shape, (2,))
        self.assertEqual(len(cutted_network), 2)

        predict = cutted_network.compile()
        x_test = asfloat(np.ones((7, 8)))
        y_predicted = predict(x_test)
        self.assertEqual(y_predicted.shape, (7, 2))

    def test_cut_input_layers_in_sequence(self):
        network = layers.join(
            layers.Input(10, name='input-1'),
            layers.Relu(5, name='relu-1'),
            layers.Relu(1, name='relu-2'),
        )

        self.assertEqual(network.input_shape, (10,))
        self.assertEqual(network.output_shape, (1,))
        self.assertEqual(len(network), 3)

        cutted_network = network.start('relu-1').start('relu-2')
        self.assertEqual(cutted_network.input_shape, (5,))
        self.assertEqual(cutted_network.output_shape, (1,))
        self.assertEqual(len(cutted_network), 1)

        predict = cutted_network.compile()
        x_test = asfloat(np.ones((7, 5)))
        y_predicted = predict(x_test)
        self.assertEqual(y_predicted.shape, (7, 1))

    def test_connect_cutted_layers_to_other_layers(self):
        network = layers.join(
            layers.Input(10, name='input-1'),
            layers.Relu(8, name='relu-0'),
            layers.Relu(5, name='relu-1'),
            layers.Relu(2, name='relu-2'),
            layers.Relu(1, name='relu-3'),
        )

        self.assertEqual(network.input_shape, (10,))
        self.assertEqual(network.output_shape, (1,))
        self.assertEqual(len(network), 5)

        cutted_network = network.start('relu-1').end('relu-2')
        self.assertEqual(cutted_network.input_shape, (8,))
        self.assertEqual(cutted_network.output_shape, (2,))
        self.assertEqual(len(cutted_network), 2)

        new_network = layers.join(
            layers.Input(8),
            cutted_network,
            layers.Sigmoid(11),
        )
        self.assertEqual(new_network.input_shape, (8,))
        self.assertEqual(new_network.output_shape, (11,))
        self.assertEqual(len(new_network), 4)

        predict = network.compile()
        x_test = asfloat(np.ones((7, 10)))
        y_predicted = predict(x_test)
        self.assertEqual(y_predicted.shape, (7, 1))

        predict = new_network.compile()
        x_test = asfloat(np.ones((7, 8)))
        y_predicted = predict(x_test)
        self.assertEqual(y_predicted.shape, (7, 11))

    def test_get_layer_by_name_from_connection(self):
        network = layers.join(
            layers.Input(10, name='input-1'),
            layers.Relu(8, name='relu-0'),
            layers.Relu(5, name='relu-1'),
        )

        reul0 = network.layer('relu-0')
        self.assertEqual(reul0.output_shape, (8,))

        reul1 = network.layer('relu-1')
        self.assertEqual(reul1.output_shape, (5,))

        with self.assertRaises(NameError):
            network.layer('some-layer-name')
