from functools import partial

from neupy import algorithms, layers

from data import simple_classification
from base import BaseTestCase


class AdagradTestCase(BaseTestCase):
    def test_simple_adagrad(self):
        x_train, x_test, y_train, y_test = simple_classification()
        mnet = algorithms.Adagrad(
            [
                layers.Input(10),
                layers.Sigmoid(20),
                layers.Sigmoid(1)
            ],
            step=0.1,
            batch_size=None,
            verbose=False,
            epsilon=1e-5,
        )
        mnet.train(x_train, y_train, x_test, y_test, epochs=100)
        self.assertGreater(0.15, mnet.validation_errors[-1])

    def test_adagrad_overfit(self):
        self.assertCanNetworkOverfit(
            partial(algorithms.Adagrad, step=0.1, verbose=False),
            epochs=3000,
        )
