import numpy as np

from neupy import layers
from neupy.layers.base import generate_layer_name
from neupy.algorithms import GradientDescent

from base import BaseTestCase


class LayersBasicsTestCase(BaseTestCase):
    def test_list_of_layers(self):
        bpnet = GradientDescent([
            layers.Input(2),
            layers.Sigmoid(3),
            layers.Sigmoid(1),
            layers.Sigmoid(10),
        ])
        self.assertEqual(
            [layer.size for layer in bpnet.layers],
            [2, 3, 1, 10]
        )

    def test_activation_layers_without_size(self):
        input_data = np.array([1, 2, -1, 10])
        expected_output = np.array([1, 2, 0, 10])

        layer = layers.Relu()
        actual_output = layer.output(input_data)

        np.testing.assert_array_equal(actual_output, expected_output)


class LayerNameTestCase(BaseTestCase):
    def test_layer_defined_name(self):
        input_layer = layers.Input(10, name='input')
        output_layer = layers.Sigmoid(1)

        layers.join(input_layer, output_layer)

        self.assertEqual(input_layer.name, 'input')

    def test_layer_default_name(self):
        input_layer = layers.Input(10)
        output_layer = layers.Sigmoid(1)

        layers.join(input_layer, output_layer)

        self.assertEqual(output_layer.name, 'sigmoid-1')
        self.assertEqual(input_layer.name, 'input-1')

    def test_layer_name_basic(self):
        name = generate_layer_name(layers.Input(10))
        self.assertEqual(name, 'input-1')

    def test_layer_name_for_connection(self):
        input_layer = layers.Input(1)
        hidden_layer = layers.Sigmoid(5)
        output_layer = layers.Sigmoid(10)

        layers.join(input_layer, hidden_layer, output_layer)

        name = generate_layer_name(hidden_layer)
        self.assertEqual(name, 'sigmoid-1')
        self.assertEqual(hidden_layer.weight.name, 'layer:sigmoid-1/weight')
        self.assertEqual(hidden_layer.bias.name, 'layer:sigmoid-1/bias')

        name = generate_layer_name(output_layer)
        self.assertEqual(name, 'sigmoid-2')
        self.assertEqual(output_layer.weight.name, 'layer:sigmoid-2/weight')
        self.assertEqual(output_layer.bias.name, 'layer:sigmoid-2/bias')

    def test_layer_name_for_connected_subgraphs(self):
        # Try to construct two separater networks and then
        # connect them into one big network
        input_layer = layers.Input(1)
        hidden1_layer = layers.Relu(2)
        conn1 = input_layer > hidden1_layer

        hidden2_layer = layers.Relu(3)
        output_layer = layers.Relu(4)
        conn2 = hidden2_layer > output_layer

        conn1 > conn2

        self.assertEqual(input_layer.name, 'input-1')
        self.assertEqual(hidden1_layer.name, 'relu-1')
        self.assertEqual(hidden2_layer.name, 'relu-2')
        self.assertEqual(output_layer.name, 'relu-3')


class InputLayerTestCase(BaseTestCase):
    def test_input_layer_exceptions(self):
        with self.assertRaises(ValueError):
            layers.Input(0)
