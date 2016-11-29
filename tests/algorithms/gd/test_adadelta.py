from neupy import algorithms

from data import simple_classification
from base import BaseTestCase


class AdadeltaTestCase(BaseTestCase):
    def test_simple_adadelta(self):
        x_train, _, y_train, _ = simple_classification()
        mnet = algorithms.Adadelta(
            (10, 20, 1),
            step=2.,
            batch_size='full',
            verbose=False,
            decay=0.95,
            epsilon=1e-5,
        )
        mnet.train(x_train, y_train, epochs=100)
        self.assertAlmostEqual(0.033, mnet.errors.last(), places=3)
