import math
import unittest

from scipy import stats
import numpy as np
import theano
import theano.tensor as T

from neupy.utils import asfloat
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
            [layer.size for layer in bpnet.layers],
            [2, 3, 1, 10]
        )

    def test_layers_iteratinos(self):
        network = GradientDescent((2, 2, 1))

        layers = list(network.layers)
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
            self.assertEqual(len(network.layers), 3)

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


class OutputLayersOperationsTestCase(BaseTestCase):
    def test_error_handling(self):
        layer = Output(1)

        with self.assertRaises(NetworkConnectionError):
            layer.relate_to(Output(1))

    def test_output_layer(self):
        layer = Output(1)
        input_vector = np.array([1, 1000, -0.1])
        output_vector = layer.output(input_vector)
        np.testing.assert_array_equal(input_vector, output_vector)

    def test_rounded_output_layer(self):
        input_vector = np.array([[1.1, 1.5, -1.99, 2]]).T

        layer = RoundedOutput(1)
        output_vector = layer.output(input_vector)
        np.testing.assert_array_equal(
            np.array([[1, 2, -2, 2]]).T,
            output_vector
        )

        layer = RoundedOutput(1, decimals=1)
        output_vector = layer.output(input_vector)
        np.testing.assert_array_equal(
            np.array([[1.1, 1.5, -2, 2]]).T,
            output_vector
        )

    def test_step_output_layer(self):
        input_vector = np.array([[-10, 0, 10, 0.001]]).T

        layer = StepOutput(1)
        output_vector = layer.output(input_vector)
        np.testing.assert_array_equal(
            np.array([[0, 0, 1, 1]]).T,
            output_vector
        )

        layer = StepOutput(1, output_bounds=(-1, 1))
        output_vector = layer.output(input_vector)
        np.testing.assert_array_equal(
            np.array([[-1, -1, 1, 1]]).T,
            output_vector
        )

        layer = StepOutput(1, critical_point=0.1)
        output_vector = layer.output(input_vector)
        np.testing.assert_array_equal(
            np.array([[0, 0, 1, 0]]).T,
            output_vector
        )

    def test_competitive_output_layer(self):
        layer = CompetitiveOutput(1)
        input_vector = np.array([[1, 10, 20, 0, -10]])
        output_vector = layer.output(input_vector)

        np.testing.assert_array_equal(
            np.array([[0, 0, 1, 0, 0]]),
            output_vector
        )

    def test_argmax_output_layer(self):
        layer = ArgmaxOutput(5)
        input_matrix = np.array([
            [1., 4, 2, 3, -10],
            [-10, 1, 0, 3.0001, 3],
            [0, 0, 0, 0, 0],
        ])
        output_vector = layer.output(input_matrix)

        np.testing.assert_array_equal(
            np.array([1, 3, 0]),
            output_vector
        )


class LayersInitializationTestCase(BaseTestCase):
    def test_layers_normal_init(self):
        input_layer = Sigmoid(30, init_method='normal')
        connection = input_layer > Output(30)
        input_layer.initialize()

        weight = input_layer.weight.get_value()
        self.assertTrue(stats.mstats.normaltest(weight))

    def test_layers_bounded_init(self):
        input_layer = Sigmoid(30, init_method='bounded',
                              bounds=(-10, 10))
        connection = input_layer > Output(10)
        input_layer.initialize()

        weight = input_layer.weight.get_value()
        self.assertLessEqual(-10, np.min(weight))
        self.assertGreaterEqual(10, np.max(weight))

    def test_layers_ortho_init(self):
        # Note: Matrix can't be orthogonal for row and column space
        # in the same time for the rectangular matrix.

        # Matrix that have more rows than columns
        input_layer = Sigmoid(30, init_method='ortho')
        connection = input_layer > Output(10)
        input_layer.initialize()

        weight = input_layer.weight.get_value()
        np.testing.assert_array_equal(
            np.eye(10),
            weight.T.dot(weight).round(10)
        )

        # Matrix that have more columns than rows
        input_layer = Sigmoid(10, init_method='ortho')
        connection = input_layer > Output(30)
        input_layer.initialize()

        weight = input_layer.weight.get_value()
        np.testing.assert_array_equal(
            np.eye(10),
            weight.dot(weight.T).round(10)
        )

    def test_he_normal(self):
        n_inputs = 30
        input_layer = Sigmoid(n_inputs, init_method='he_normal')
        connection = input_layer > Output(30)
        input_layer.initialize()

        weight = input_layer.weight.get_value()

        self.assertAlmostEqual(weight.mean(), 0, places=1)
        self.assertAlmostEqual(weight.std(), math.sqrt(2. / n_inputs),
                               places=2)
        self.assertTrue(stats.mstats.normaltest(weight))

    def test_he_uniform(self):
        n_inputs = 10
        input_layer = Sigmoid(n_inputs, init_method='he_uniform')
        connection = input_layer > Output(30)
        input_layer.initialize()

        weight = input_layer.weight.get_value()
        bound = math.sqrt(6. / n_inputs)

        self.assertAlmostEqual(weight.mean(), 0, places=1)
        self.assertGreaterEqual(weight.min(), -bound)
        self.assertLessEqual(weight.max(), bound)

    def test_xavier_normal(self):
        n_inputs, n_outputs = 30, 30
        input_layer = Sigmoid(n_inputs, init_method='xavier_normal')
        connection = input_layer > Output(n_outputs)
        input_layer.initialize()

        weight = input_layer.weight.get_value()

        self.assertAlmostEqual(weight.mean(), 0, places=1)
        self.assertAlmostEqual(weight.std(),
                               math.sqrt(2. / (n_inputs + n_outputs)),
                               places=2)
        self.assertTrue(stats.mstats.normaltest(weight))

    def test_xavier_uniform(self):
        n_inputs, n_outputs = 10, 30
        input_layer = Sigmoid(n_inputs, init_method='xavier_uniform')
        connection = input_layer > Output(n_outputs)
        input_layer.initialize()

        weight = input_layer.weight.get_value()
        bound = math.sqrt(6. / (n_inputs + n_outputs))

        self.assertAlmostEqual(weight.mean(), 0, places=1)
        self.assertGreaterEqual(weight.min(), -bound)
        self.assertLessEqual(weight.max(), bound)
