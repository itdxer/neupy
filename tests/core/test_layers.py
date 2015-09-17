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
        layer1 = SigmoidLayer(1)
        layer2 = SigmoidLayer(1, function_coef={'alpha': 0.1})
        self.assertNotEqual(layer1.activation_function(1),
                            layer2.activation_function(1))

    def test_step_layer(self):
        layer1 = StepLayer(1)
        self.assertEqual(layer1.activation_function(1).item(0), 1)
        self.assertEqual(layer1.activation_function(-1).item(0), 0)

    def test_linear_layer(self):
        layer = LinearLayer(1)
        self.assertEqual(layer.activation_function(1), 1)

    def test_tanh_layer(self):
        layer1 = TanhLayer(1)
        layer2 = TanhLayer(1, function_coef={'alpha': 0.5})
        self.assertGreater(layer1.activation_function(1),
                           layer2.activation_function(1))

    def test_rectifier_layer(self):
        layer = RectifierLayer(1)
        self.assertEqual(0, layer.activation_function(-10))
        self.assertEqual(0, layer.activation_function(0))
        self.assertEqual(10, layer.activation_function(10))

    def test_softplus_layer(self):
        layer = SoftplusLayer(1)
        self.assertEqual(math.log(2), layer.activation_function(0))

    def test_softmax_layer(self):
        test_input = np.array([[0.5, 0.5, 0.1]])

        layer1 = SoftmaxLayer(3)
        test_correct_result = np.array([[0.37448695, 0.37448695, 0.25102611]])
        self.assertTrue(np.allclose(
            test_correct_result, layer1.activation_function(test_input)
        ))

        layer2 = SoftmaxLayer(3, function_coef={'temp': 100000000})
        self.assertTrue(np.allclose(
            np.ones(3) / 3., layer2.activation_function(test_input)
        ))

        layer3 = SoftmaxLayer(3, function_coef={'temp': 0.01})
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
        layer = OutputLayer(1)

        with self.assertRaises(NetworkConnectionError):
            layer.relate_to(OutputLayer(1))

        self.assertEqual(layer.format_output(1), 1)
        self.assertEqual(layer.output(1), 1)

        layer = RoundOutputLayer(1)
        self.assertEqual(layer.format_output(1.1), 1)

        layer = StepOutputLayer(1)
        self.assertEqual(layer.format_output(-10), 0)

        layer = CompetitiveOutputLayer(1)
        output = np.sum(layer.format_output(np.array([[1, 10, 20, 0, -10]])))
        self.assertEqual(output, 1)

    @unittest.skip("Not ready yet")
    def test_recurrent_connections(self):
        inp = SigmoidLayer(2)
        hd = [SigmoidLayer(2), SigmoidLayer(2)]
        out = OutputLayer(1)

        network = Backpropagation(
            connection=(
                inp > hd[0] > out,
                      hd[0] > hd[1],
                              hd[1] > hd[0],
            )
        )

    def test_layers_init_method(self):
        input_layer = SigmoidLayer(30, init_method='gauss')
        connection = input_layer > OutputLayer(10)
        input_layer.initialize()
        self.assertTrue(stats.mstats.normaltest(input_layer.weight))

        input_layer = SigmoidLayer(30, init_method='bounded',
                                   random_weight_bound=(-10, 10))
        connection = input_layer > OutputLayer(10)
        input_layer.initialize()
        self.assertLessEqual(-10, np.min(input_layer.weight))
        self.assertGreaterEqual(10, np.max(input_layer.weight))

        input_layer = SigmoidLayer(30, init_method='ortho',
                                   random_weight_bound=(-10, 10))
        connection = input_layer > OutputLayer(10)
        input_layer.initialize()
        weight = input_layer.weight
        # Can't be orthogonal in both ways for rectangular matrix.
        self.assertEqualArrays(np.eye(10), weight.T.dot(weight).round(10))

        input_layer = SigmoidLayer(10, init_method='ortho',
                                   random_weight_bound=(-10, 10))
        connection = input_layer > OutputLayer(30)
        input_layer.initialize()
        weight = input_layer.weight
        self.assertEqualArrays(np.eye(10), weight.dot(weight.T).round(10))

    def test_without_output_layer(self):
        with self.assertRaises(NetworkConnectionError):
            network = Backpropagation(
                connection=(
                    layers.SigmoidLayer(10),
                    layers.SigmoidLayer(1),
                )
            )

    def test_list_of_layers(self):
        bpnet = Backpropagation([SigmoidLayer(2), SigmoidLayer(3),
                                 SigmoidLayer(1), OutputLayer(10)])
        self.assertEqual([layer.input_size for layer in bpnet.layers],
                         [2, 3, 1, 10])
