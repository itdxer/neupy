import math

from scipy import stats
import numpy as np

from neupy.layers import *

from base import BaseTestCase


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
        np.testing.assert_array_almost_equal(
            np.eye(10),
            weight.T.dot(weight),
            decimal=5
        )

        # Matrix that have more columns than rows
        input_layer = Sigmoid(10, init_method='ortho')
        connection = input_layer > Output(30)
        input_layer.initialize()

        weight = input_layer.weight.get_value()
        np.testing.assert_array_almost_equal(
            np.eye(10),
            weight.dot(weight.T),
            decimal=5
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
