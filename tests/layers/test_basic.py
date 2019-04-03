import copy

import numpy as np
import tensorflow as tf

from neupy import layers, algorithms, init
from neupy.utils import asfloat
from neupy.exceptions import LayerConnectionError

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

    def test_layer_definitions(self):
        Conv = layers.Convolution.define(
            padding='SAME',
            weight=init.Constant(1),
            bias=None,
        )
        network = layers.join(
            layers.Input((28, 28, 1)),
            Conv((3, 3, 16)),
            Conv((3, 3, 32)),
        )
        network.create_variables()

        self.assertShapesEqual(network.output_shape, (None, 28, 28, 32))

        weight_1 = self.eval(network.layers[1].weight)
        self.assertEqual(weight_1.sum(), 1 * 3 * 3 * 16)
        self.assertIsNone(network.layers[1].bias)

        weight_2 = self.eval(network.layers[2].weight)
        self.assertEqual(weight_2.sum(), 16 * 3 * 3 * 32)
        self.assertIsNone(network.layers[2].bias)


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


class LayerCopyTestCase(BaseTestCase):
    def test_layer_copy(self):
        relu = layers.Relu(10, weight=init.Normal(), bias=None)
        copied_relu = copy.copy(relu)

        self.assertEqual(relu.name, 'relu-1')
        self.assertEqual(copied_relu.name, 'relu-2')

        self.assertIsInstance(relu.weight, init.Normal)
        self.assertIsNone(relu.bias)

    def test_initialized_layer_copy(self):
        network = layers.Input(10) >> layers.Relu(5)
        network.create_variables()

        relu = network.layers[1]
        copied_relu = copy.copy(relu)

        self.assertEqual(relu.name, 'relu-1')
        self.assertEqual(copied_relu.name, 'relu-2')
        self.assertIsInstance(copied_relu.weight, np.ndarray)

        error_message = (
            "Cannot connect layer `relu-1` to layer `relu-2`, because output "
            "shape \(\(\?, 5\)\) of the first layer is incompatible with the "
            "input shape \(\(\?, 10\)\) of the second layer."
        )
        with self.assertRaisesRegexp(LayerConnectionError, error_message):
            # copied relu expects 10 input features, but network outputs 5
            layers.join(network, copied_relu)

    def test_layer_deep_copy(self):
        relu = layers.Relu(10, weight=np.zeros((5, 10)))

        copied_relu = copy.copy(relu)
        self.assertIs(relu.weight, copied_relu.weight)

        deepcopied_relu = copy.deepcopy(relu)
        self.assertIsNot(relu.weight, deepcopied_relu.weight)
