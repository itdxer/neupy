import numpy as np

from neupy import layers
from neupy.utils import asfloat, as_tuple

from base import BaseTestCase


class OldInlineDefinitionsTestCase(BaseTestCase):
    def test_inline_connection_python_compatibility(self):
        connection = layers.Input(1) > layers.Relu(2)
        self.assertTrue(connection.__bool__())
        self.assertTrue(connection.__nonzero__())

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

    def test_inline_connection_wtih_different_pointers(self):
        relu_2 = layers.Relu(2)

        connection_1 = layers.Input(1) > relu_2 > layers.Relu(3)
        connection_2 = relu_2 > layers.Relu(4)
        connection_3 = layers.Input(1) > relu_2

        self.assertShapesEqual(connection_1.input_shape, (None, 1))
        self.assertShapesEqual(connection_1.output_shape, (None, 3))

        self.assertShapesEqual(connection_2.input_shape, None)
        self.assertShapesEqual(connection_2.output_shape, (None, 4))

        self.assertShapesEqual(connection_3.input_shape, (None, 1))
        self.assertShapesEqual(connection_3.output_shape, (None, 2))

        self.assertIn(relu_2, connection_2.input_layers)
        self.assertIn(relu_2, connection_3.output_layers)

    def test_inline_connection_with_parallel_connection(self):
        left_branch = layers.join(
            layers.Convolution((3, 3, 32)),
            layers.Relu(),
            layers.MaxPooling((2, 2)),
        )

        right_branch = layers.join(
            layers.Convolution((7, 7, 16)),
            layers.Relu(),
        )

        input_layer = layers.Input((10, 10, 3))
        concat = layers.Concatenate()

        network_concat = input_layer > (left_branch | right_branch) > concat
        network = network_concat > layers.Reshape() > layers.Softmax()

        self.assertShapesEqual(network_concat.input_shape, (None, 10, 10, 3))
        self.assertShapesEqual(network_concat.output_shape, (None, 4, 4, 48))

        self.assertShapesEqual(network.input_shape, (None, 10, 10, 3))
        self.assertShapesEqual(network.output_shape, (None, 768))


class InlineDefinitionsTestCase(BaseTestCase):
    def test_inline_definition(self):
        connection = layers.Input(2) >> layers.Relu(10) >> layers.Tanh(1)
        self.assertShapesEqual(connection.input_shape, (None, 2))
        self.assertShapesEqual(connection.output_shape, (None, 1))

        input_value = asfloat(np.random.random((10, 2)))
        output_value = self.eval(connection.output(input_value))
        self.assertEqual(output_value.shape, (10, 1))

    def test_connection_shape_multiple_inputs(self):
        in1 = layers.Input(10)
        in2 = layers.Input(20)
        conn = (in1 | in2) >> layers.Concatenate()

        self.assertShapesEqual(conn.input_shape, [(None, 10), (None, 20)])
        self.assertShapesEqual(conn.output_shape, (None, 30))

    def test_connection_shape_multiple_outputs(self):
        conn = layers.Input(10) >> (layers.Sigmoid(1) | layers.Sigmoid(2))
        self.assertShapesEqual(conn.input_shape, (None, 10))
        self.assertShapesEqual(conn.output_shape, [(None, 1), (None, 2)])

    def test_inline_operation_order(self):
        in1 = layers.Input(10)
        out1 = layers.Sigmoid(1)
        out2 = layers.Sigmoid(2)
        conn = in1 >> out1 | out2

        self.assertShapesEqual(conn.input_shape, [(None, 10), None])
        self.assertShapesEqual(conn.output_shape, [(None, 1), (None, 2)])

    def test_sequential_partial_definitions(self):
        # Tree structure:
        #
        #                       Sigmoid(10)
        #                      /
        # Input(10) - Sigmoid(5)
        #                      \
        #                       Softmax(10)
        #
        input_layer = layers.Input(10)
        minimized = input_layer >> layers.Sigmoid(5)
        reconstructed = minimized >> layers.Sigmoid(10)
        classifier = minimized >> layers.Softmax(20)

        x_matrix = asfloat(np.random.random((3, 10)))
        minimized_output = self.eval(minimized.output(x_matrix))
        self.assertEqual((3, 5), minimized_output.shape)

        reconstructed_output = self.eval(reconstructed.output(x_matrix))
        self.assertEqual((3, 10), reconstructed_output.shape)

        classifier_output = self.eval(classifier.output(x_matrix))
        self.assertEqual((3, 20), classifier_output.shape)

    def test_inplace_seq_operator(self):
        network = layers.Input(1)
        network >>= layers.Relu(2)
        network >>= layers.Relu(3)

        self.assertEqual(len(network), 3)
        self.assertShapesEqual(network.input_shape, (None, 1))
        self.assertShapesEqual(network.output_shape, (None, 3))


class DefinitionsTestCase(BaseTestCase):
    def test_one_to_many_parallel_connection_output(self):
        one_to_many = layers.join(
            layers.Input(4),
            layers.parallel(
                layers.Linear(11),
                layers.Linear(12),
                layers.Linear(13),
            ),
        )

        input_value = asfloat(np.random.random((10, 4)))
        actual_output = self.eval(one_to_many.output(input_value))

        self.assertEqual(actual_output[0].shape, (10, 11))
        self.assertEqual(actual_output[1].shape, (10, 12))
        self.assertEqual(actual_output[2].shape, (10, 13))

    def test_connections_with_complex_parallel_relations(self):
        input_layer = layers.Input((5, 5, 3))
        network = layers.join(
            layers.parallel([
                layers.Convolution((1, 1, 8)),
            ], [
                layers.Convolution((1, 1, 4)),
                layers.parallel(
                    layers.Convolution((1, 3, 2), padding='same'),
                    layers.Convolution((3, 1, 2), padding='same'),
                ),
            ], [
                layers.Convolution((1, 1, 8)),
                layers.Convolution((3, 3, 4), padding='same'),
                layers.parallel(
                    layers.Convolution((1, 3, 2), padding='same'),
                    layers.Convolution((3, 1, 2), padding='same'),
                )
            ], [
                layers.MaxPooling((3, 3), padding='same', stride=(1, 1)),
                layers.Convolution((1, 1, 8)),
            ]),
            layers.Concatenate(),
        )
        self.assertShapesEqual(network.input_shape, [None, None, None, None])
        self.assertShapesEqual(network.output_shape, (None, None, None, None))

        # Connect them at the end, because we need to make
        # sure tha parallel networks defined without input shapes
        network = layers.join(input_layer, network)
        self.assertShapesEqual(network.output_shape, (None, 5, 5, 24))

    def test_residual_connections(self):
        network = layers.join(
            layers.Input((5, 5, 3)),
            layers.parallel(
                layers.Identity(),
                layers.join(
                    layers.Convolution((3, 3, 8), padding='same'),
                    layers.Relu(),
                ),
            ),
            layers.Concatenate(),
        )

        self.assertShapesEqual((None, 5, 5, 3), network.input_shape)
        self.assertShapesEqual((None, 5, 5, 11), network.output_shape)
