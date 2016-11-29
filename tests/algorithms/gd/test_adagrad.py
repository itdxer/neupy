from neupy import algorithms

from data import simple_classification
from base import BaseTestCase


class AdagradTestCase(BaseTestCase):
    def test_simple_adagrad(self):
        x_train, _, y_train, _ = simple_classification()
        mnet = algorithms.Adagrad(
            (10, 20, 1),
            step=2.,
            batch_size='full',
            verbose=False,
            epsilon=1e-5,
        )
        mnet.train(x_train, y_train, epochs=100)
        self.assertAlmostEqual(0.068, mnet.errors.last(), places=3)
