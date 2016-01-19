from functools import partial
import numpy as np

from neupy import algorithms

from data import simple_classification
from utils import compare_networks
from base import BaseTestCase


class MomentumTestCase(BaseTestCase):
    def test_basic(self):
        x_train, _, y_train, _ = simple_classification()
        mnet = algorithms.Momentum(
            (10, 20, 1),
            step=0.35,
            momentum=0.25,
            batch_size=1,
            verbose=False
        )

        mnet.train(x_train, y_train, epochs=40)
        self.assertAlmostEqual(0.009, mnet.last_error(), places=3)

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
           show_comparison_plot=False
        )
