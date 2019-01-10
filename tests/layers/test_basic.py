import numpy as np
import tensorflow as tf

from neupy import layers, algorithms
from neupy.utils import asfloat

from base import BaseTestCase


class LayersBasicsTestCase(BaseTestCase):
    def test_shared_parameters_between_layers(self):
        weight = tf.Variable(
            np.random.random((10, 10)),
            name='weight',
            dtype=tf.float32)

        inp = layers.Input(10)
        hid1 = layers.Relu(10, weight=weight)
        hid2 = layers.Relu(10, weight=weight)

        network = inp >> hid1 >> hid2
        self.assertIs(hid1.weight, hid2.weight)

        # Check that it is able to train network without errors
        x_train = y_train = asfloat(np.random.random((15, 10)))
        gdnet = algorithms.GradientDescent(network)
        gdnet.train(x_train, y_train, epochs=5)

        np.testing.assert_array_almost_equal(
            self.eval(hid1.weight),
            self.eval(hid2.weight),
        )

    def test_predict_new_model(self):
        input_value = np.random.random((7, 4))

        network = layers.Input(4) >> layers.Relu(5)
        output_value = network.predict(input_value)

        self.assertEqual(output_value.shape, (7, 5))


class LayerNameTestCase(BaseTestCase):
    def test_layer_defined_name(self):
        input_layer = layers.Input(10, name='input')
        self.assertEqual(input_layer.name, 'input')

    def test_layer_default_name(self):
        input_layer = layers.Input(10)
        output_layer = layers.Sigmoid(1)

        self.assertEqual(output_layer.name, 'sigmoid-1')
        self.assertEqual(input_layer.name, 'input-1')

    def test_layer_name_for_network(self):
        input_layer = layers.Input(1)
        hidden_layer = layers.Sigmoid(5)
        output_layer = layers.Sigmoid(10)

        network = input_layer >> hidden_layer >> output_layer
        network.outputs

        self.assertEqual(hidden_layer.name, 'sigmoid-1')
        self.assertIn('layer/sigmoid-1/weight', hidden_layer.weight.name)
        self.assertIn('layer/sigmoid-1/bias', hidden_layer.bias.name)

        self.assertEqual(output_layer.name, 'sigmoid-2')
        self.assertIn('layer/sigmoid-2/weight', output_layer.weight.name)
        self.assertIn('layer/sigmoid-2/bias', output_layer.bias.name)

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
            def output(self, input):
                return input

        layer = ABCD()
        self.assertEqual(layer.name, 'abcd-1')

    def test_new_layer_init_exception(self):
        with self.assertRaisesRegexp(TypeError, "abstract methods output"):
            class NewLayer(layers.BaseLayer):
                pass

            NewLayer()

    def test_layer_name_with_first_few_capital_letters(self):
        class ABCDef(layers.BaseLayer):
            def output(self, input):
                return input

        layer_1 = ABCDef()
        self.assertEqual(layer_1.name, 'abc-def-1')

        class abcDEF(layers.BaseLayer):
            def output(self, input):
                return input

        layer_2 = abcDEF()
        self.assertEqual(layer_2.name, 'abcdef-1')

        class abcDef(layers.BaseLayer):
            def output(self, input):
                return input

        layer_3 = abcDef()
        self.assertEqual(layer_3.name, 'abc-def-1')
