from neupy import algorithms

from data import simple_classification
from base import BaseTestCase


class RMSPropTestCase(BaseTestCase):
    def test_simple_rmsprop(self):
        x_train, _, y_train, _ = simple_classification()
        mnet = algorithms.RMSProp(
            (10, 20, 1),
            step=.1,
            batch_size='full',
            verbose=False,
            epsilon=1e-5,
            decay=0.9,
        )
        mnet.train(x_train, y_train, epochs=100)
        self.assertAlmostEqual(0.01, mnet.errors.last(), places=2)
