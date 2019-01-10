from functools import partial

from neupy import algorithms, layers

from helpers import simple_classification
from helpers import compare_networks
from base import BaseTestCase


class MomentumTestCase(BaseTestCase):
    def test_simple_momentum(self):
        x_train, x_test, y_train, y_test = simple_classification()
        mnet = algorithms.Momentum(
            [
                layers.Input(10),
                layers.Sigmoid(20),
                layers.Sigmoid(1)
            ],
            step=0.35,
            momentum=0.99,
            batch_size=None,
            verbose=False,
            nesterov=True,
        )

        mnet.train(x_train, y_train, x_test, y_test, epochs=30)
        self.assertGreater(0.15, mnet.validation_errors[-1])

    def test_momentum_with_minibatch(self):
        x_train, _, y_train, _ = simple_classification()
        compare_networks(
           # Test classes
           partial(algorithms.Momentum, batch_size=None),
           partial(algorithms.Momentum, batch_size=1),
           # Test data
           (x_train, y_train),
           # Network configurations
           network=[
               layers.Input(10),
               layers.Sigmoid(20),
               layers.Sigmoid(1)
           ],
           step=0.25,
           momentum=0.1,
           shuffle_data=True,
           verbose=False,
           # Test configurations
           epochs=40,
           show_comparison_plot=False,
        )

    def test_nesterov_momentum(self):
        x_train, _, y_train, _ = simple_classification()
        compare_networks(
           # Test classes
           partial(algorithms.Momentum, nesterov=False),
           partial(algorithms.Momentum, nesterov=True),
           # Test data
           (x_train, y_train),
           # Network configurations
           network=[
               layers.Input(10),
               layers.Sigmoid(20),
               layers.Sigmoid(1)
           ],
           batch_size=None,
           step=0.25,
           momentum=0.9,
           shuffle_data=True,
           verbose=False,
           # Test configurations
           epochs=10,
           show_comparison_plot=False,
        )

    def test_momentum_overfit(self):
        self.assertCanNetworkOverfit(
            partial(algorithms.Momentum, step=0.3, verbose=False),
            epochs=1500,
        )

    def test_nesterov_momentum_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.Momentum,
                step=0.3,
                nesterov=True,
                verbose=False,
            ),
            epochs=1500,
        )
