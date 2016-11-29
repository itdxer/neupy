from neupy import algorithms

from data import simple_classification
from base import BaseTestCase


class AdamaxTestCase(BaseTestCase):
    def test_simple_adamax(self):
        x_train, _, y_train, _ = simple_classification()
        mnet = algorithms.Adamax(
            (10, 20, 1),
            step=.01,
            batch_size='full',
            verbose=False,
            epsilon=1e-8,
            beta1=0.9,
            beta2=0.999,
        )
        mnet.train(x_train, y_train, epochs=100)
        self.assertAlmostEqual(0.038, mnet.errors.last(), places=3)
