import theano.tensor as T

from neupy import layers
from neupy.layers.connections import NetworkConnectionError
from neupy.network.constructor import (generate_layers, create_input_variable,
                                       create_output_variable,
                                       ConstructableNetwork)

from base import BaseTestCase


class NetworkConstructorTestCase(BaseTestCase):
    def test_generate_layers_expcetion(self):
        with self.assertRaises(ValueError):
            generate_layers((5,))

    def test_invalid_dim_for_input_layer(self):
        with self.assertRaises(ValueError):
            create_input_variable(layers.Input((1, 2, 3, 4)), name='test')

    def test_custom_output_variable(self):
        def error_func(expected, predicted):
            return T.abs(expected - predicted)

        error_func.expected_dtype = T.vector
        var = create_output_variable(error_func, name='test_func')

        self.assertIn('vector', str(var.type))


class ConstructableNetworkTestCase(BaseTestCase):
    def test_multi_input_exception(self):
        input_layer_1 = layers.Input(10)
        input_layer_2 = layers.Input(10)
        output_layer = layers.Concatenate()

        connection = layers.join(input_layer_1, output_layer)
        connection = layers.join(input_layer_2, output_layer)

        with self.assertRaises(NetworkConnectionError):
            ConstructableNetwork(connection)

    def test_multi_output_exception(self):
        input_layer = layers.Input(10)
        output_layer_1 = layers.Sigmoid(20)
        output_layer_2 = layers.Sigmoid(30)

        connection = layers.join(input_layer, output_layer_1)
        connection = layers.join(input_layer, output_layer_2)

        with self.assertRaises(NetworkConnectionError):
            ConstructableNetwork(connection)

    def test_no_updates_by_default(self):
        net = ConstructableNetwork(layers.Input(10) > layers.Sigmoid(1))
        updates = net.init_param_updates(layer=None, parameter=None)
        self.assertEqual(updates, [])
