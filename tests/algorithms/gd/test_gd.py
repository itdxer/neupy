from functools import partial

import numpy as np

from neupy import algorithms, layers, environment
from neupy.algorithms.gd.base import apply_batches

from utils import compare_networks
from base import BaseTestCase
from data import simple_classification


class GradientDescentTestCase(BaseTestCase):
    def test_network_attrs(self):
        network = algorithms.GradientDescent((2, 2, 1), verbose=False)
        network.step = 0.1
        network.error = 'mse'
        network.shuffle_data = True

        with self.assertRaises(TypeError):
            network.step = '33'

        with self.assertRaises(ValueError):
            network.error = 'not a function'

        with self.assertRaises(TypeError):
            network.shuffle_data = 1

    def test_gd(self):
        environment.reproducible()
        x_train, _, y_train, _ = simple_classification()

        network = algorithms.GradientDescent(
            layers.Input(10) > layers.Tanh(20) > layers.Tanh(1),
            step=0.1,
            verbose=False
        )
        network.train(x_train, y_train, epochs=100)
        self.assertLess(network.errors.last(), 0.05)

    def test_addons_exceptions(self):
        with self.assertRaises(ValueError):
            # Invalid optimization class
            algorithms.GradientDescent(
                (2, 3, 1),
                addons=[algorithms.GradientDescent]
            )

        with self.assertRaises(ValueError):
            # Dublicate optimization algorithms from one type
            algorithms.GradientDescent(
                (2, 3, 1),
                addons=[algorithms.WeightDecay, algorithms.WeightDecay],
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
            addons=[algorithms.WeightDecay, algorithms.SearchThenConverge],
            verbose=False
        )

    def test_minibatch_gd(self):
        x_train, _, y_train, _ = simple_classification()
        compare_networks(
           # Test classes
           partial(algorithms.GradientDescent, verbose=False),
           partial(algorithms.MinibatchGradientDescent,
                   batch_size=1, verbose=False),
           # Test data
           (x_train, y_train),
           # Network configurations
           connection=(layers.Input(10) > layers.Tanh(20) > layers.Tanh(1)),
           step=0.02,
           shuffle_data=True,
           verbose=False,
           # Test configurations
           epochs=40,
           show_comparison_plot=False
        )

    def test_gd_get_params_method(self):
        network = algorithms.GradientDescent((2, 3, 1))

        self.assertIn('connection', network.get_params(with_connection=True))
        self.assertNotIn(
            'connection',
            network.get_params(with_connection=False),
        )

    def test_gd_overfit(self):
        self.assertCanNetworkOverfit(
            partial(algorithms.GradientDescent, step=1.0, verbose=False),
            epochs=4000,
        )

    def test_gd_minibatch_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.MinibatchGradientDescent,
                step=0.5,
                batch_size=5,
                verbose=False,
            ),
            epochs=4000,
        )


class GDAdditionalFunctionsTestCase(BaseTestCase):
    def test_gd_apply_batches_exceptions(self):
        with self.assertRaisesRegexp(ValueError, "at least one element"):
            apply_batches(function=lambda x: x, arguments=[], batch_size=12)

        with self.assertRaisesRegexp(ValueError, "Cannot show error"):
            apply_batches(
                function=lambda x: x,
                arguments=[np.random.random((36, 1))],
                batch_size=12,
                show_error_output=True,
                scalar_output=False,
            )

        with self.assertRaisesRegexp(ValueError, "Cannot convert output"):
            apply_batches(
                function=lambda x: x,
                arguments=[np.random.random((36, 1))],
                batch_size=12,
            )
