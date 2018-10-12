from functools import partial

from neupy import algorithms

from data import simple_classification
from base import BaseTestCase


class AdamTestCase(BaseTestCase):
    def test_simple_adam(self):
        x_train, x_test, y_train, y_test = simple_classification()
        mnet = algorithms.Adam(
            (10, 20, 1),
            step=0.1,
            verbose=False,
            epsilon=1e-4,
            beta1=0.9,
            beta2=0.99,
        )
        mnet.train(x_train, y_train, x_test, y_test, epochs=200)
        self.assertGreater(0.15, mnet.validation_errors.last())

    def test_adam_overfit(self):
        self.assertCanNetworkOverfit(
            partial(algorithms.Adam, step=0.1, verbose=False),
            epochs=500,
        )
