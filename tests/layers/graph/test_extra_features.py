from neupy import layers

from base import BaseTestCase


class GraphExtraFeaturesTestCase(BaseTestCase):
    def test_single_input_for_parallel_layers(self):
        left = layers.Input(10, name='input') >> layers.Sigmoid(5)
        right = left >> layers.Sigmoid(2)
        network = (left | right) >> layers.Concatenate()

        self.assertEqual(len(network.input_layers), 1)

        input_layer = network.input_layers[0]
        self.assertEqual(input_layer.name, 'input')

    def test_single_output_for_parallel_layers(self):
        left = layers.Sigmoid(5, name='output')
        right = layers.Relu(10) >> left
        network = layers.Input(10) >> (left | right)

        self.assertEqual(len(network.output_layers), 1)

        output_layer = network.output_layers[0]
        self.assertEqual(output_layer.name, 'output')
