from functools import partial

from neupy import algorithms

from utils import compare_networks
from data import simple_classification
from base import BaseTestCase


class HessianTestCase(BaseTestCase):
    def test_hessian_exceptions(self):
        with self.assertRaises(ValueError):
            # Don't have step option
            algorithms.Hessian((2, 3, 1), step=1)

    def test_compare_bp_and_hessian(self):
        x_train, x_test, y_train, y_test = simple_classification()
        compare_networks(
            # Test classes
            algorithms.GradientDescent,
            partial(algorithms.Hessian, inv_penalty_const=1),
            # Test data
            (x_train, y_train, x_test, y_test),
            # Network configurations
            connection=(10, 15, 1),
            shuffle_data=True,
            verbose=False,
            show_epoch=1,
            # Test configurations
            epochs=5,
            show_comparison_plot=False
        )
