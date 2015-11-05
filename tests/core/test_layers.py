import math
import unittest

from scipy import stats
import numpy as np

from neupy.algorithms import Backpropagation
from neupy.network.connections import NetworkConnectionError
from neupy.layers import *

from base import BaseTestCase


class LayersTestCase(BaseTestCase):
    verbose = False

    def test_layers_iteratinos(self):
        network = Backpropagation((2, 2, 1))

        layers = list(network.layers)
        output_layer = layers.pop()

        self.assertIsNone(output_layer.relate_to_layer)

        for layer in layers:
            self.assertIsNotNone(layer.relate_to_layer)

    def test_sigmoid_layer(self):
        layer1 = Sigmoid(1)
        layer2 = Sigmoid(1, function_coef={'alpha': 0.1})
        self.assertNotEqual(layer1.activation_function(1),
                            layer2.activation_function(1))

    def test_step_layer(self):
        layer1 = Step(1)
        self.assertEqual(layer1.activation_function(1).item(0), 1)
        self.assertEqual(layer1.activation_function(-1).item(0), 0)

    def test_linear_layer(self):
        layer = Linear(1)
        self.assertEqual(layer.activation_function(1), 1)

    def test_tanh_layer(self):
        layer1 = Tanh(1)
        layer2 = Tanh(1, function_coef={'alpha': 0.5})
        self.assertGreater(layer1.activation_function(1),
                           layer2.activation_function(1))

    def test_rectifier_layer(self):
        layer = Rectifier(1)
        self.assertEqual(0, layer.activation_function(-10))
        self.assertEqual(0, layer.activation_function(0))
        self.assertEqual(10, layer.activation_function(10))

    def test_softplus_layer(self):
        layer = Softplus(1)
        self.assertEqual(math.log(2), layer.activation_function(0))

    def test_softmax_layer(self):
        test_input = np.array([[0.5, 0.5, 0.1]])

        layer1 = Softmax(3)
        test_correct_result = np.array([[0.37448695, 0.37448695, 0.25102611]])
        self.assertTrue(np.allclose(
            test_correct_result, layer1.activation_function(test_input)
        ))

        layer2 = Softmax(3, function_coef={'temp': 100000000})
        self.assertTrue(np.allclose(
            np.ones(3) / 3., layer2.activation_function(test_input)
        ))

        layer3 = Softmax(3, function_coef={'temp': 0.01})
        test_correct_result = np.array([[0.5, 0.5, 0.0]])
        self.assertTrue(np.allclose(
            test_correct_result, layer3.activation_function(test_input)
        ))

    def test_euclide_distance_layer(self):
        layer = EuclideDistanceLayer(2, weight=np.array([[0.4, 1]]))
        test_correct_result = np.array([[-0.72111026, -1.]])
        self.assertTrue(np.allclose(
            test_correct_result, layer.output(np.array([[0, 1]]))
        ))

    def test_cosine_distance_layer(self):
        layer = AngleDistanceLayer(2, weight=np.array([[0.4, 1]]).T)
        test_correct_result = np.array([[-0.40489179]])
        self.assertTrue(np.allclose(
            test_correct_result, layer.output(np.array([[0.1, 0.1]]))
        ))

    def test_output_layers(self):
        layer = Output(1)

        with self.assertRaises(NetworkConnectionError):
            layer.relate_to(Output(1))

        self.assertEqual(layer.format_output(1), 1)
        self.assertEqual(layer.output(1), 1)

        layer = RoundedOutput(1)
        self.assertEqual(layer.format_output(1.1), 1)

        layer = StepOutput(1)
        self.assertEqual(layer.format_output(-10), 0)

        layer = CompetitiveOutput(1)
        output = np.sum(layer.format_output(np.array([[1, 10, 20, 0, -10]])))
        self.assertEqual(output, 1)

    @unittest.skip("Not ready yet")
    def test_recurrent_connections(self):
        inp = Sigmoid(2)
        hd = [Sigmoid(2), Sigmoid(2)]
        out = Output(1)

        network = Backpropagation(
            connection=(
                inp > hd[0] > out,
                      hd[0] > hd[1],
                              hd[1] > hd[0],
            )
        )

    def test_layers_init_method(self):
        input_layer = Sigmoid(30, init_method='gauss')
        connection = input_layer > Output(10)
        input_layer.initialize()
        self.assertTrue(stats.mstats.normaltest(input_layer.weight))

        input_layer = Sigmoid(30, init_method='bounded',
                                   bounds=(-10, 10))
        connection = input_layer > Output(10)
        input_layer.initialize()
        self.assertLessEqual(-10, np.min(input_layer.weight))
        self.assertGreaterEqual(10, np.max(input_layer.weight))

        input_layer = Sigmoid(30, init_method='ortho',
                                   bounds=(-10, 10))
        connection = input_layer > Output(10)
        input_layer.initialize()
        weight = input_layer.weight
        # Can't be orthogonal in both ways for rectangular matrix.
        np.testing.assert_array_equal(
            np.eye(10),
            weight.T.dot(weight).round(10)
        )

        input_layer = Sigmoid(10, init_method='ortho',
                                   bounds=(-10, 10))
        connection = input_layer > Output(30)
        input_layer.initialize()
        weight = input_layer.weight
        np.testing.assert_array_equal(
            np.eye(10),
            weight.dot(weight.T).round(10)
        )

    def test_without_output_layer(self):
        with self.assertRaises(NetworkConnectionError):
            network = Backpropagation(
                connection=(
                    layers.Sigmoid(10),
                    layers.Sigmoid(1),
                )
            )

    def test_list_of_layers(self):
        bpnet = Backpropagation([Sigmoid(2), Sigmoid(3),
                                 Sigmoid(1), Output(10)])
        self.assertEqual([layer.input_size for layer in bpnet.layers],
                         [2, 3, 1, 10])
