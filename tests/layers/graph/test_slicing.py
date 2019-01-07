import numpy as np

from neupy import layers
from neupy.utils import asfloat

from base import BaseTestCase


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

        x_test = asfloat(np.ones((7, 10)))
        y_predicted = self.eval(relu_1_network.output(x_test))
        self.assertEqual(y_predicted.shape, (7, 5))

    def test_select_network_branch(self):
        network = layers.join(
            layers.Input(10, name='input-1'),
            layers.parallel(
                layers.Relu(1, name='relu-1'),
                layers.Relu(2, name='relu-2'),
            )
        )

        self.assertEqual(network.input_shape, (10,))
        self.assertEqual(network.output_shape, [(1,), (2,)])
        self.assertEqual(len(network), 3)

        relu_1_network = network.end('relu-1')
        self.assertEqual(relu_1_network.input_shape, (10,))
        self.assertEqual(relu_1_network.output_shape, (1,))
        self.assertEqual(len(relu_1_network), 2)

        x_test = asfloat(np.ones((7, 10)))
        y_predicted = self.eval(relu_1_network.output(x_test))
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

        x_test = asfloat(np.ones((7, 10)))
        y_predicted = cutted_network.output(x_test)
        self.assertEqual(y_predicted.shape, (7, 10))

    def test_cut_using_layer_object(self):
        relu = layers.Relu(2)
        network = layers.Input(10) >> relu >> layers.Sigmoid(1)

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
        self.assertEqual(relu_1_network.input_shape, None)
        self.assertEqual(relu_1_network.output_shape, (1,))
        self.assertEqual(len(relu_1_network), 2)
        self.assertDictEqual(relu_1_network.forward_graph, {
            network.layer('relu-1'): [network.layer('relu-2')],
            network.layer('relu-2'): [],
        })

        x_test = asfloat(np.ones((7, 10)))
        y_predicted = self.eval(relu_1_network.output(x_test))
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

        self.assertEqual(cutted_network.input_shape, None)
        self.assertEqual(cutted_network.output_shape, (2,))
        self.assertEqual(len(cutted_network), 2)
        self.assertDictEqual(cutted_network.forward_graph, {
            network.layer('relu-1'): [network.layer('relu-2')],
            network.layer('relu-2'): [],
        })

        x_test = asfloat(np.ones((7, 8)))
        y_predicted = self.eval(cutted_network.output(x_test))
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
        self.assertEqual(cutted_network.input_shape, None)
        self.assertEqual(cutted_network.output_shape, (1,))
        self.assertEqual(len(cutted_network), 1)
        self.assertDictEqual(cutted_network.forward_graph, {
            network.layer('relu-2'): [],
        })

        x_test = asfloat(np.ones((7, 5)))
        y_predicted = self.eval(cutted_network.output(x_test))
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
        self.assertEqual(cutted_network.input_shape, None)
        self.assertEqual(cutted_network.output_shape, (2,))
        self.assertEqual(len(cutted_network), 2)
        self.assertDictEqual(cutted_network.forward_graph, {
            network.layer('relu-1'): [network.layer('relu-2')],
            network.layer('relu-2'): [],
        })

        new_network = layers.join(
            layers.Input(8),
            cutted_network,
            layers.Sigmoid(11),
        )
        self.assertEqual(new_network.input_shape, (8,))
        self.assertEqual(new_network.output_shape, (11,))
        self.assertEqual(len(new_network), 4)

        x_test = asfloat(np.ones((7, 10)))
        y_predicted = self.eval(network.output(x_test))
        self.assertEqual(y_predicted.shape, (7, 1))

        x_test = asfloat(np.ones((7, 8)))
        y_predicted = self.eval(new_network.output(x_test))
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
