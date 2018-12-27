from functools import partial

from neupy import algorithms, layers

from data import simple_classification
from base import BaseTestCase


class AdadeltaTestCase(BaseTestCase):
    def test_simple_adadelta(self):
        x_train, x_test, y_train, y_test = simple_classification()
        mnet = algorithms.Adadelta(
            [
                layers.Input(10),
                layers.Sigmoid(20),
                layers.Sigmoid(1)
            ],
            batch_size=None,
            verbose=False,
            decay=0.95,
            epsilon=1e-5,
        )
        mnet.train(x_train, y_train, x_test, y_test, epochs=100)
        self.assertGreater(0.05, mnet.training_errors[-1])
        self.assertGreater(0.15, mnet.validation_errors[-1])

    def test_adadelta_overfit(self):
        self.assertCanNetworkOverfit(
            partial(algorithms.Adadelta, verbose=False),
            epochs=3000)
