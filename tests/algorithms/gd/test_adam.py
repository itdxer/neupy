from neupy import algorithms

from data import simple_classification
from base import BaseTestCase


class AdamTestCase(BaseTestCase):
    def test_simple_adam(self):
        x_train, _, y_train, _ = simple_classification()
        mnet = algorithms.Adam(
            (10, 20, 1),
            step=15.,
            batch_size='full',
            verbose=False,
            epsilon=1e-8,
            beta1=0.9,
            beta2=0.999,
        )
        mnet.train(x_train, y_train, epochs=100)
        self.assertAlmostEqual(0.06, mnet.errors.last(), places=2)
