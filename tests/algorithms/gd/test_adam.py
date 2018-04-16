from neupy import algorithms

from data import simple_classification
from base import BaseTestCase


class AdamTestCase(BaseTestCase):
    def test_simple_adam(self):
        x_train, x_test, y_train, y_test = simple_classification()
        mnet = algorithms.Adam(
            (10, 20, 1),
            step=10.,
            verbose=False,
            epsilon=1e-7,
            beta1=0.9,
            beta2=0.999,
        )
        mnet.train(x_train, y_train, x_test, y_test, epochs=100)
        self.assertAlmostEqual(0.08, mnet.errors.last(), places=2)
