from neupy import algorithms

from data import simple_classification
from base import BaseTestCase


class AdagradTestCase(BaseTestCase):
    def test_simple_adagrad(self):
        x_train, x_test, y_train, y_test = simple_classification()
        mnet = algorithms.Adagrad(
            (10, 20, 1),
            step=2.,
            batch_size='full',
            verbose=False,
            epsilon=1e-5,
        )
        mnet.train(x_train, y_train, x_test, y_test, epochs=100)
        self.assertGreater(0.15, mnet.validation_errors.last())
