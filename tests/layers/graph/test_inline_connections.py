from neupy import layers

from base import BaseTestCase


class InlineConnectionsTestCase(BaseTestCase):
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

        network_concat = input_layer >> (left_branch | right_branch) >> concat
        network = network_concat >> layers.Reshape() >> layers.Softmax()

        self.assertEqual(network_concat.input_shape, (10, 10, 3))
        self.assertEqual(network_concat.output_shape, (4, 4, 48))

        self.assertEqual(network.input_shape, (10, 10, 3))
        self.assertEqual(network.output_shape, (768,))

    def test_inline_connection_wtih_different_pointers(self):
        relu_2 = layers.Relu(2)

        connection_1 = layers.Input(1) >> relu_2 >> layers.Relu(3)
        connection_2 = relu_2 >> layers.Relu(4)
        connection_3 = layers.Input(1) >> relu_2

        self.assertEqual(connection_1.input_shape, (1,))
        self.assertEqual(connection_1.output_shape, (3,))

        self.assertEqual(connection_2.input_shape, (1,))
        self.assertEqual(connection_2.output_shape, (4,))

        self.assertEqual(connection_3.input_shape, (1,))
        self.assertEqual(connection_3.output_shape, (2,))

        self.assertIn(relu_2, connection_2.input_layers)
        self.assertIn(relu_2, connection_3.output_layers)

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

    def test_repeated_inline_connections(self):
        input_layer_1 = layers.Input(1)
        input_layer_2 = layers.Input(1)
        hidden_layer = layers.Relu(17)
        output_layer = layers.Softmax(5)

        connection_1 = input_layer_1 > hidden_layer > output_layer
        connection_2 = input_layer_2 > hidden_layer > output_layer

        self.assertListEqual(
            list(connection_1),
            [input_layer_1, hidden_layer, output_layer])

        self.assertListEqual(
            list(connection_2),
            [input_layer_2, hidden_layer, output_layer])

    def test_repeated_inline_connections_with_list(self):
        input_layer_1 = layers.Input(1)
        input_layer_2 = layers.Input(1)
        hd1 = layers.Relu(4)
        hd2 = layers.Sigmoid(4)
        output_layer = layers.Softmax(5)

        connection_1 = input_layer_1 > [hd1, hd2] > output_layer
        connection_2 = input_layer_2 > [hd1, hd2] > output_layer

        self.assertListEqual(
            list(connection_1),
            [input_layer_1, hd1, hd2, output_layer])

        self.assertListEqual(
            list(connection_2),
            [input_layer_2, hd1, hd2, output_layer])

    def test_inline_connection_python_compatibility(self):
        connection = layers.Input(1) > layers.Relu(2)
        self.assertTrue(connection.__bool__())
        self.assertTrue(connection.__nonzero__())

    def test_repeated_reverse_inline_connection(self):
        input_layer_1 = layers.Input(1)
        input_layer_2 = layers.Input(1)
        hidden_layer = layers.Relu(4)
        output_layer = layers.Softmax(5)

        connection_1 = output_layer < hidden_layer < input_layer_1
        connection_2 = output_layer < hidden_layer < input_layer_2

        self.assertListEqual(
            list(connection_1),
            [input_layer_1, hidden_layer, output_layer])

        self.assertListEqual(
            list(connection_2),
            [input_layer_2, hidden_layer, output_layer])

    def test_mixed_inline_connections_many_in_one_out(self):
        input_layer_1 = layers.Input(1)
        input_layer_2 = layers.Input(1)
        output_layer = layers.Sigmoid(5)

        connection = input_layer_1 > output_layer < input_layer_2

        self.assertEqual(len(connection), 3)
        self.assertEqual(connection.input_shape, [(1,), (1,)])
        self.assertEqual(connection.output_shape, (5,))

    def test_mixed_inline_connections_one_in_many_out(self):
        input_layer = layers.Input(2)
        output_layer_1 = layers.Sigmoid(10)
        output_layer_2 = layers.Sigmoid(20)

        connection = output_layer_1 < input_layer > output_layer_2

        self.assertEqual(len(connection), 3)
        self.assertEqual(connection.input_shape, (2,))
        self.assertEqual(connection.output_shape, [(20,), (10,)])

    def test_mixed_inline_connections_many_in_many_out(self):
        in1 = layers.Input(1)
        in2 = layers.Input(1)
        hd = layers.Relu(5)
        out1 = layers.Sigmoid(10)
        out2 = layers.Sigmoid(20)

        connection = in1 > out1 < in2 > hd > out2

        self.assertEqual(len(connection), 5)
        self.assertEqual(connection.input_shape, [(1,), (1,)])
        self.assertEqual(connection.output_shape, [(20,), (10,)])
