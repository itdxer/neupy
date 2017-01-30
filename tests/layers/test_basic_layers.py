import numpy as np

from neupy import layers, algorithms

from base import BaseTestCase


class LayersBasicsTestCase(BaseTestCase):
    def test_shared_parameters_between_layers(self):
        hidden_layer_1 = layers.Relu(10)
        network = layers.Input(10) > hidden_layer_1

        hidden_layer_2 = layers.Relu(10, weight=hidden_layer_1.weight,
                                     bias=hidden_layer_1.bias)
        network = network > hidden_layer_2

        self.assertIs(hidden_layer_1.weight, hidden_layer_2.weight)
        self.assertIs(hidden_layer_1.bias, hidden_layer_2.bias)

        # Check that it is able to train network without errors
        x_train = y_train = np.random.random((15, 10))
        gdnet = algorithms.GradientDescent(network)
        gdnet.train(x_train, y_train, epochs=5)

        np.testing.assert_array_almost_equal(
            hidden_layer_1.weight.get_value(),
            hidden_layer_2.weight.get_value(),
        )
        np.testing.assert_array_almost_equal(
            hidden_layer_1.bias.get_value(),
            hidden_layer_2.bias.get_value(),
        )


class LayerNameTestCase(BaseTestCase):
    def test_layer_defined_name(self):
        input_layer = layers.Input(10, name='input')
        self.assertEqual(input_layer.name, 'input')

    def test_layer_default_name(self):
        input_layer = layers.Input(10)
        output_layer = layers.Sigmoid(1)

        self.assertEqual(output_layer.name, 'sigmoid-1')
        self.assertEqual(input_layer.name, 'input-1')

    def test_layer_name_for_connection(self):
        input_layer = layers.Input(1)
        hidden_layer = layers.Sigmoid(5)
        output_layer = layers.Sigmoid(10)

        layers.join(input_layer, hidden_layer, output_layer)

        self.assertEqual(hidden_layer.name, 'sigmoid-1')
        self.assertEqual(hidden_layer.weight.name, 'layer:sigmoid-1/weight')
        self.assertEqual(hidden_layer.bias.name, 'layer:sigmoid-1/bias')

        self.assertEqual(output_layer.name, 'sigmoid-2')
        self.assertEqual(output_layer.weight.name, 'layer:sigmoid-2/weight')
        self.assertEqual(output_layer.bias.name, 'layer:sigmoid-2/bias')

    def test_layer_name_with_repeated_layer_type(self):
        input_layer = layers.Input(1)
        hidden1_layer = layers.Relu(2)
        hidden2_layer = layers.Relu(3)
        output_layer = layers.Relu(4)

        self.assertEqual(input_layer.name, 'input-1')
        self.assertEqual(hidden1_layer.name, 'relu-1')
        self.assertEqual(hidden2_layer.name, 'relu-2')
        self.assertEqual(output_layer.name, 'relu-3')

    def test_layer_name_with_capital_letters(self):
        class ABCD(layers.BaseLayer):
            pass

        layer = ABCD()
        self.assertEqual(layer.name, 'abcd-1')

    def test_layer_name_with_first_few_capital_letters(self):
        class ABCDef(layers.BaseLayer):
            pass

        layer_1 = ABCDef()
        self.assertEqual(layer_1.name, 'abc-def-1')

        class abcDEF(layers.BaseLayer):
            pass

        layer_2 = abcDEF()
        self.assertEqual(layer_2.name, 'abcdef-1')

        class abcDef(layers.BaseLayer):
            pass

        layer_3 = abcDef()
        self.assertEqual(layer_3.name, 'abc-def-1')


class InputLayerTestCase(BaseTestCase):
    def test_input_layer_exceptions(self):
        with self.assertRaises(ValueError):
            layers.Input(0)
