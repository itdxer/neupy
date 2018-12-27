from functools import partial

from neupy import algorithms, layers, utils

from helpers import compare_networks
from base import BaseTestCase
from helpers import simple_classification


class GradientDescentTestCase(BaseTestCase):
    def test_network_attrs(self):
        network = algorithms.GradientDescent(
            [
                layers.Input(2),
                layers.Sigmoid(2),
                layers.Sigmoid(1),
            ],
            verbose=False,
        )
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
        utils.reproducible()
        x_train, _, y_train, _ = simple_classification()

        network = algorithms.GradientDescent(
            layers.Input(10) > layers.Tanh(20) > layers.Tanh(1),
            step=0.1,
            verbose=False
        )
        network.train(x_train, y_train, epochs=100)
        self.assertLess(network.training_errors[-1], 0.05)

    def test_minibatch_gd(self):
        x_train, _, y_train, _ = simple_classification()
        compare_networks(
           # Test classes
           partial(algorithms.GradientDescent, verbose=False),
           partial(algorithms.GradientDescent,
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
        network = algorithms.GradientDescent(
            layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1))

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
                algorithms.GradientDescent,
                step=0.5,
                batch_size=5,
                verbose=False,
            ),
            epochs=4000,
        )


class GDAdditionalFunctionsTestCase(BaseTestCase):
    def test_small_network_representation(self):
        network = algorithms.GradientDescent(
            layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1))
        self.assertIn("Input(2) > Sigmoid(3) > Sigmoid(1)", str(network))

    def test_large_network_representation(self):
        network = algorithms.GradientDescent([
            layers.Input(1),
            layers.Sigmoid(1),
            layers.Sigmoid(1),
            layers.Sigmoid(1),
            layers.Sigmoid(1),
            layers.Sigmoid(1),
        ])
        self.assertIn("[... 6 layers ...]", str(network))
