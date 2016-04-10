import math
import unittest

from scipy import stats
import numpy as np
import theano
import theano.tensor as T

from neupy.utils import asfloat
from neupy import layers
from neupy.algorithms import GradientDescent
from neupy.layers.connections import NetworkConnectionError
from neupy.layers import *

from base import BaseTestCase


class LayersBasicsTestCase(BaseTestCase):
    def test_without_output_layer(self):
        with self.assertRaises(NetworkConnectionError):
            GradientDescent(layers.Sigmoid(10) > layers.Sigmoid(1))

    def test_list_of_layers(self):
        bpnet = GradientDescent([Sigmoid(2), Sigmoid(3),
                                 Sigmoid(1), Output(10)])
        self.assertEqual(
            [layer.size for layer in bpnet.all_layers],
            [2, 3, 1, 10]
        )

    def test_layers_iteratinos(self):
        network = GradientDescent((2, 2, 1))

        layers = list(network.all_layers)
        output_layer = layers.pop()

        self.assertIsNone(output_layer.relate_to_layer)
        for layer in layers:
            self.assertIsNotNone(layer.relate_to_layer)

    def test_connection_initializations(self):
        possible_connections = (
            (2, 3, 1),
            [Sigmoid(2), Tanh(3), Output(1)],
            Relu(2) > Tanh(10) > Output(1),
        )

        for connection in possible_connections:
            network = GradientDescent(connection)
            self.assertEqual(len(network.all_layers), 3)

    @unittest.skip("Not ready yet")
    def test_recurrent_connections(self):
        inp = Sigmoid(2)
        hd = [Sigmoid(2), Sigmoid(2)]
        out = Output(1)

        GradientDescent(
            connection=(
                inp > hd[0] > out,
                      hd[0] > hd[1],
                              hd[1] > hd[0],
            )
        )

    def test_activation_layers_without_size(self):
        input_data = np.array([1, 2, -1, 10])
        expected_output = np.array([1, 2, 0, 10])

        layer = layers.Relu()
        actual_output = layer.output(input_data)

        np.testing.assert_array_equal(actual_output, expected_output)



class HiddenLayersOperationsTestCase(BaseTestCase):
    def test_sigmoid_layer(self):
        layer1 = Sigmoid(1)
        self.assertGreater(1, layer1.activation_function(1).eval())

    def test_hard_sigmoid_layer(self):
        layer1 = HardSigmoid(6)

        test_value = asfloat(np.array([[-3, -2, -1, 0, 1, 2]]))
        expected = np.array([[0, 0.1, 0.3, 0.5, 0.7, 0.9]])

        x = T.matrix()
        output = layer1.activation_function(x).eval({x: test_value})

        np.testing.assert_array_almost_equal(output, expected)

    def test_step_layer(self):
        layer1 = Step(1)

        input_vector = theano.shared(np.array([-10, -1, 0, 1, 10]))
        expected = np.array([0, 0, 0, 1, 1])
        output = layer1.activation_function(input_vector).eval()
        np.testing.assert_array_equal(output, expected)

    def test_linear_layer(self):
        layer = Linear(1)
        self.assertEqual(layer.activation_function(1), 1)

    def test_tanh_layer(self):
        layer1 = Tanh(1)
        self.assertGreater(1, layer1.activation_function(1).eval())

    def test_relu_layer(self):
        layer = Relu(1)
        self.assertEqual(0, layer.activation_function(-10))
        self.assertEqual(0, layer.activation_function(0))
        self.assertEqual(10, layer.activation_function(10))

    def test_softplus_layer(self):
        layer = Softplus(1)
        self.assertAlmostEqual(
            math.log(2),
            layer.activation_function(0).eval()
        )

    def test_softmax_layer(self):
        test_input = np.array([[0.5, 0.5, 0.1]])

        softmax_layer = Softmax(3)
        correct_result = np.array([[0.37448695, 0.37448695, 0.25102611]])
        np.testing.assert_array_almost_equal(
            correct_result,
            softmax_layer.activation_function(test_input).eval()
        )

    def test_dropout_layer(self):
        test_input = np.ones((50, 20))
        dropout_layer = Dropout(proba=0.5)

        layer_output = dropout_layer.output(test_input).eval()

        self.assertGreater(layer_output.sum(), 900)
        self.assertLess(layer_output.sum(), 1100)

        self.assertTrue(np.all(
            np.bitwise_or(layer_output == 0, layer_output == 2)
        ))

    def test_reshape_layer(self):
        x = np.random.random((5, 4, 3, 2, 1))
        reshape_layer = Reshape()
        y = reshape_layer.output(x).eval()
        self.assertEqual(y.shape, (5, 4 * 3 * 2 * 1))
