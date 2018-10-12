from functools import partial

from neupy import algorithms

from data import simple_classification
from base import BaseTestCase


class AdadeltaTestCase(BaseTestCase):
    def test_simple_adadelta(self):
        x_train, x_test, y_train, y_test = simple_classification()
        mnet = algorithms.Adadelta(
            (10, 20, 1),
            batch_size='full',
            verbose=False,
            decay=0.95,
            epsilon=1e-5,
        )
        mnet.train(x_train, y_train, x_test, y_test, epochs=100)
        self.assertGreater(0.05, mnet.errors.last())
        self.assertGreater(0.15, mnet.validation_errors.last())

    def test_adadelta_overfit(self):
        self.assertCanNetworkOverfit(
            partial(algorithms.Adadelta, verbose=False),
            epochs=3000,
        )
