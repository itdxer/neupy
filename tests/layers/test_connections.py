import unittest

import numpy as np
import theano
import theano.tensor as T

from neupy import layers, algorithms
from neupy.utils import asfloat, as_tuple
from neupy.layers import Input, Relu, Tanh, Sigmoid
from neupy.layers.connections import (is_sequential, merge_dicts_with_list,
                                      does_layer_expect_one_input)

from base import BaseTestCase


class ConnectionsTestCase(BaseTestCase):
    def test_connection_initializations(self):
        possible_connections = (
            (2, 3, 1),
            [Input(2), Sigmoid(3), Tanh(1)],
            Input(2) > Relu(10) > Tanh(1),
            Tanh(1) < Relu(10) < Input(2),
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
        connection.initialize()

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
        connection.initialize()

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


class ConnectionTypesTestCase(BaseTestCase):
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

        reconstructed.initialize()
        classifier.initialize()

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

        minimized.initialize()
        reconstructed.initialize()

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
        connection = layers.Parallel(
            [[
                layers.Convolution((8, 1, 1)),
            ], [
                layers.Convolution((4, 1, 1)),
                layers.Parallel(
                    [[

                        layers.Convolution((2, 1, 3), padding=(0, 1)),
                    ], [
                        layers.Convolution((2, 3, 1), padding=(1, 0)),
                    ]],
                    layers.Concatenate(),
                )
            ], [
                layers.Convolution((8, 1, 1)),
                layers.Convolution((4, 3, 3), padding=1),
                layers.Parallel(
                    [[

                        layers.Convolution((2, 1, 3), padding=(0, 1)),
                    ], [
                        layers.Convolution((2, 3, 1), padding=(1, 0)),
                    ]],
                    layers.Concatenate(),
                )
            ], [
                layers.MaxPooling((3, 3), stride=(1, 1), padding=1),
                layers.Convolution((8, 1, 1)),
            ]],
            layers.Concatenate(),
        )
        # Connect them at the end, because we need to make
        # sure tha parallel connections defined without
        # input shapes
        connection = input_layer > connection
        self.assertEqual((24, 5, 5), connection.output_shape)


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

    @unittest.skip("broken")
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
