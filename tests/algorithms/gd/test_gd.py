from functools import partial

import numpy as np

from neupy import algorithms, layers

from utils import compare_networks
from base import BaseTestCase
from data import simple_classification


class GradientDescentTestCase(BaseTestCase):
    def test_network_attrs(self):
        network = algorithms.GradientDescent((2, 2, 1), verbose=False)
        network.step = 0.1
        network.bias = True
        network.error = 'mse'
        network.shuffle_data = True

        with self.assertRaises(TypeError):
            network.step = '33'

        with self.assertRaises(ValueError):
            network.error = 'not a function'

        with self.assertRaises(TypeError):
            network.shuffle_data = 1

    def test_gd(self):
        x_train, _, y_train, _ = simple_classification()

        network = algorithms.GradientDescent(
            (layers.Tanh(10) > layers.Tanh(20) > layers.Output(1)),
            step=0.3,
            verbose=False
        )
        network.train(x_train, y_train, epochs=500)
        self.assertAlmostEqual(network.errors.last(), 0.02, places=3)

    def test_optimization_validations(self):
        with self.assertRaises(ValueError):
            # Invalid optimization class
            algorithms.GradientDescent(
                (2, 3, 1),
                addons=[algorithms.GradientDescent]
            )

        with self.assertRaises(ValueError):
            # Dublicate optimization algorithms from one type
            algorithms.GradientDescent(
                (2, 3, 1), addons=[algorithms.WeightDecay,
                                          algorithms.WeightDecay]
            )

        algorithms.GradientDescent(
            (2, 3, 1),
            addons=[algorithms.WeightDecay],
            verbose=False,
        )
        algorithms.GradientDescent(
            (2, 3, 1),
            addons=[algorithms.SearchThenConverge],
            verbose=False,
        )
        algorithms.GradientDescent(
            (2, 3, 1),
            addons=[algorithms.WeightDecay,
                           algorithms.SearchThenConverge],
            verbose=False
        )

    def test_minibatch_gd(self):
        x_train, _, y_train, _ = simple_classification()
        compare_networks(
           # Test classes
           algorithms.GradientDescent,
           partial(algorithms.MinibatchGradientDescent, batch_size=1),
           # Test data
           (x_train, y_train),
           # Network configurations
           connection=(layers.Tanh(10) > layers.Tanh(20) > layers.Output(1)),
           step=0.1,
           shuffle_data=True,
           verbose=False,
           # Test configurations
           epochs=40,
           show_comparison_plot=False
        )
