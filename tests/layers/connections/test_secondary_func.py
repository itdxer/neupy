from neupy import layers
from neupy.layers.connections.base import is_sequential
from neupy.layers.connections.graph import does_layer_expect_one_input

from base import BaseTestCase


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
