import numpy as np

from neupy import layers
from neupy.utils import asfloat, as_tuple

from base import BaseTestCase


class ConnectionTypesTestCase(BaseTestCase):
    def test_inline_connections(self):
        input_layer = layers.Input(784)
        conn = input_layer > layers.Sigmoid(20)
        conn = conn > layers.Sigmoid(10)

        self.assertEqual(3, len(conn))

        in_sizes = [784, 784, 20]
        out_sizes = [784, 20, 10]

        for layer, in_size, out_size in zip(conn, in_sizes, out_sizes):
            self.assertEqual(
                layer.input_shape, as_tuple(in_size),
                msg="Layer: {}".format(layer))

            self.assertEqual(
                layer.output_shape, as_tuple(out_size),
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

        x_matrix = asfloat(np.random.random((3, 10)))
        minimized_output = self.eval(minimized.output(x_matrix))
        self.assertEqual((3, 5), minimized_output.shape)

        reconstructed_output = self.eval(reconstructed.output(x_matrix))
        self.assertEqual((3, 10), reconstructed_output.shape)

        classifier_output = self.eval(classifier.output(x_matrix))
        self.assertEqual((3, 20), classifier_output.shape)

    def test_dict_based_inputs_into_connection_with_layer_names(self):
        encoder = layers.Input(10) > layers.Sigmoid(5, name='sigmoid-1')
        decoder = layers.Sigmoid(2, name='sigmoid-2') > layers.Sigmoid(10)

        network = encoder > decoder

        x_test = asfloat(np.ones((7, 5)))
        y_predicted = self.eval(network.output({'sigmoid-2': x_test}))

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

        x_matrix = asfloat(np.random.random((3, 10)))
        minimized_output = self.eval(minimized.output(x_matrix))
        self.assertEqual((3, 5), minimized_output.shape)

        reconstructed_output = self.eval(
            reconstructed.output({output_layer: minimized_output})
        )
        self.assertEqual((3, 10), reconstructed_output.shape)

    def test_connections_with_complex_parallel_relations(self):
        input_layer = layers.Input((5, 5, 3))
        connection = layers.join(
            [[
                layers.Convolution((1, 1, 8)),
            ], [
                layers.Convolution((1, 1, 4)),
                [[
                    layers.Convolution((1, 3, 2), padding='SAME'),
                ], [
                    layers.Convolution((3, 1, 2), padding='SAME'),
                ]],
            ], [
                layers.Convolution((1, 1, 8)),
                layers.Convolution((3, 3, 4), padding='SAME'),
                [[
                    layers.Convolution((1, 3, 2), padding='SAME'),
                ], [
                    layers.Convolution((3, 1, 2), padding='SAME'),
                ]],
            ], [
                layers.MaxPooling((3, 3), padding='SAME', stride=(1, 1)),
                layers.Convolution((1, 1, 8)),
            ]],
            layers.Concatenate(),
        )
        self.assertEqual(connection.input_shape, [None, None, None, None])

        # Connect them at the end, because we need to make
        # sure tha parallel connections defined without
        # input shapes
        connection = input_layer > connection
        self.assertEqual((5, 5, 24), connection.output_shape)

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
