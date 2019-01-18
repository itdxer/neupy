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

    def test_check_if_network_sequential(self):
        network = layers.join(
            layers.Input(10),
            layers.Relu(5),
            layers.Relu(3),
        )
        self.assertTrue(network.is_sequential())

        network = layers.join(
            layers.Input(10),
            layers.parallel(
                layers.Relu(5),
                layers.Relu(3),
            ),
            layers.Concatenate(),
        )
        self.assertFalse(network.is_sequential())

        network = layers.parallel(
            layers.Relu(5),
            layers.Relu(3),
        )
        self.assertFalse(network.is_sequential())

    def test_empty_graph(self):
        graph = layers.LayerGraph()
        self.assertEqual(len(graph), 0)
        self.assertEqual(str(graph), "[empty graph]")
