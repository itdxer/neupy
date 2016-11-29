from functools import partial

from neupy import algorithms

from data import simple_classification
from utils import compare_networks
from base import BaseTestCase


class MomentumTestCase(BaseTestCase):
    def test_simple_momentum(self):
        x_train, _, y_train, _ = simple_classification()
        mnet = algorithms.Momentum(
            (10, 20, 1),
            step=0.35,
            momentum=0.99,
            batch_size='full',
            verbose=False,
            nesterov=True,
        )

        mnet.train(x_train, y_train, epochs=40)
        self.assertAlmostEqual(0.017, mnet.errors.last(), places=3)

    def test_with_minibatch(self):
        x_train, _, y_train, _ = simple_classification()
        compare_networks(
           # Test classes
           partial(algorithms.Momentum, batch_size='full'),
           partial(algorithms.Momentum, batch_size=1),
           # Test data
           (x_train, y_train),
           # Network configurations
           connection=(10, 20, 1),
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
           connection=(10, 20, 1),
           batch_size='full',
           step=0.25,
           momentum=0.9,
           shuffle_data=True,
           verbose=False,
           # Test configurations
           epochs=10,
           show_comparison_plot=False,
        )
