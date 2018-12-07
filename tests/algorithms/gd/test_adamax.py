from functools import partial

from neupy import algorithms

from data import simple_classification
from base import BaseTestCase


class AdamaxTestCase(BaseTestCase):
    def test_simple_adamax(self):
        x_train, x_test, y_train, y_test = simple_classification()
        mnet = algorithms.Adamax(
            (10, 20, 1),
            step=0.1,
            batch_size='full',
            verbose=False,
            epsilon=1e-7,
            beta1=0.9,
            beta2=0.999,
        )
        mnet.train(x_train, y_train, x_test, y_test, epochs=50)
        self.assertGreater(0.15, mnet.errors.last())

    def test_adamax_overfit(self):
        self.assertCanNetworkOverfit(
            partial(algorithms.Adamax, verbose=False),
            epochs=2500,
        )
