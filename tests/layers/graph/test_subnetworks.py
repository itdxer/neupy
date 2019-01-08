import numpy as np

from neupy import layers, algorithms
from neupy.utils import asfloat

from base import BaseTestCase


class SubnetworksTestCase(BaseTestCase):
    def test_subnetwork_in_mlp_network(self):
        network = layers.join(
            layers.Input(2),
            layers.Relu(10),
            layers.Relu(4) >> layers.Relu(7),
            layers.Relu(3) >> layers.Relu(1),
        )

        self.assertEqual(len(network), 6)
        self.assertTrue(network.is_sequential())
        self.assertShapesEqual(network.input_shape, (None, 2))
        self.assertShapesEqual(network.output_shape, (None, 1))

    def test_subnetwork_in_conv_network(self):
        network = layers.join(
            layers.Input((28, 28, 1)),

            layers.Convolution((3, 3, 8)) >> layers.Relu(),
            layers.Convolution((3, 3, 8)) >> layers.Relu(),
            layers.MaxPooling((2, 2)),

            layers.Reshape(),
            layers.Softmax(1),
        )

        self.assertEqual(8, len(network))
        self.assertTrue(network.is_sequential())
        self.assertShapesEqual(network.input_shape, (None, 28, 28, 1))
        self.assertShapesEqual(network.output_shape, (None, 1))

        expected_order = [
            layers.Input,
            layers.Convolution, layers.Relu,
            layers.Convolution, layers.Relu,
            layers.MaxPooling, layers.Reshape, layers.Softmax,
        ]
        for actual_layer, expected_layer in zip(network, expected_order):
            self.assertIsInstance(actual_layer, expected_layer)

    def test_many_to_many_parallel_subnetworks(self):
        connection = layers.parallel(
            layers.Input(1) >> layers.Linear(11),
            layers.Input(2) >> layers.Linear(12),
            layers.Input(3) >> layers.Linear(13),
        )

        input_value_1 = asfloat(np.random.random((10, 1)))
        input_value_2 = asfloat(np.random.random((20, 2)))
        input_value_3 = asfloat(np.random.random((30, 3)))

        actual_output = self.eval(
            connection.output(input_value_1, input_value_2, input_value_3))

        self.assertEqual(actual_output[0].shape, (10, 11))
        self.assertEqual(actual_output[1].shape, (20, 12))
        self.assertEqual(actual_output[2].shape, (30, 13))
