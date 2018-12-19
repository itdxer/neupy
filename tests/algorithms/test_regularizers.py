import numpy as np

from neupy import algorithms, layers

from base import BaseTestCase


class L2RegularizationTestCase(BaseTestCase):
    def test_l2_regularization(self):
        network = layers.Input(10) > layers.Relu(5, weight=2, bias=2)
        regularizer = algorithms.l2(0.01, exclude=['bias'])

        regularization_cost = self.eval(regularizer(network))
        self.assertAlmostEqual(regularization_cost, 2.0)

    def test_l2_regularization_with_bias(self):
        network = layers.Input(10) > layers.Relu(5, weight=2, bias=2)
        regularizer = algorithms.l2(0.01, exclude=[])

        regularization_cost = self.eval(regularizer(network))
        self.assertAlmostEqual(regularization_cost, 2.2)

    def test_l2_repr(self):
        l2_repr = repr(algorithms.l2(0.01, exclude=['bias']))
        self.assertEqual(l2_repr, "l2(0.01, exclude=['bias'])")

        l2_repr = repr(algorithms.l2(decay_rate=0.01, exclude=['bias']))
        self.assertEqual(l2_repr, "l2(decay_rate=0.01, exclude=['bias'])")


class L1RegularizationTestCase(BaseTestCase):
    def test_l1_regularization(self):
        weight = 2 * np.sign(np.random.random((10, 5)) - 0.5)
        network = layers.Input(10) > layers.Relu(5, weight=weight, bias=2)
        regularizer = algorithms.l1(0.01)

        regularization_cost = self.eval(regularizer(network))
        self.assertAlmostEqual(regularization_cost, 1.0)


class MaxNormRegularizationTestCase(BaseTestCase):
    def test_max_norm_regularization(self):
        weight = np.arange(20).reshape(4, 5)
        network = layers.Input(4) > layers.Relu(5, weight=weight, bias=100)
        regularizer = algorithms.maxnorm(0.01)

        regularization_cost = self.eval(regularizer(network))
        self.assertAlmostEqual(regularization_cost, 0.19)
