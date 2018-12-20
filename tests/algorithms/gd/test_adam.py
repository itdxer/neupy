from functools import partial

from neupy import algorithms, layers

from data import simple_classification
from base import BaseTestCase


class AdamTestCase(BaseTestCase):
    def test_simple_adam(self):
        x_train, x_test, y_train, y_test = simple_classification()
        mnet = algorithms.Adam(
            [
                layers.Input(10),
                layers.Sigmoid(20),
                layers.Sigmoid(1)
            ],
            step=0.1,
            verbose=True,
            epsilon=1e-4,
            beta1=0.9,
            beta2=0.99,
        )
        mnet.train(x_train, y_train, x_test, y_test, epochs=200)
        self.assertGreater(0.2, mnet.validation_errors[-1])

    def test_adam_overfit(self):
        self.assertCanNetworkOverfit(
            partial(algorithms.Adam, step=0.1, verbose=False),
            epochs=500,
        )
