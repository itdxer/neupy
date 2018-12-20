from functools import partial

from neupy import algorithms, layers

from data import simple_classification
from base import BaseTestCase


class RMSPropTestCase(BaseTestCase):
    def test_simple_rmsprop(self):
        x_train, x_test, y_train, y_test = simple_classification()
        mnet = algorithms.RMSProp(
            [
                layers.Input(10),
                layers.Sigmoid(20),
                layers.Sigmoid(1)
            ],
            step=0.02,
            batch_size='full',
            verbose=False,
            epsilon=1e-5,
            decay=0.9,
        )
        mnet.train(x_train, y_train, x_test, y_test, epochs=100)
        self.assertGreater(0.11, mnet.validation_errors.last())

    def test_rmsprop_overfit(self):
        self.assertCanNetworkOverfit(
            partial(algorithms.RMSProp, step=0.01, verbose=False),
            epochs=2200,
        )
