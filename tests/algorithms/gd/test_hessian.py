from functools import partial

import theano
import theano.tensor as T
import numpy as np

from neupy import algorithms
from neupy.algorithms.gd.hessian import find_hessian_and_gradient

from utils import compare_networks
from data import simple_classification
from base import BaseTestCase


class HessianTestCase(BaseTestCase):
    # In case of Hessian this solution will give
    # significant improvement.
    use_sandbox_mode = False

    def test_hessian_exceptions(self):
        with self.assertRaises(ValueError):
            # Don't have step parameter
            algorithms.Hessian((2, 3, 1), step=1)

    def test_compare_bp_and_hessian(self):
        x_train, x_test, y_train, y_test = simple_classification()
        compare_networks(
            # Test classes
            algorithms.GradientDescent,
            partial(algorithms.Hessian, penalty_const=1),
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

    def test_hessian_computation(self):
        x = T.scalar('x')
        y = T.scalar('y')

        f = x ** 2 + y ** 3 + 7 * x * y
        # Gradient function:
        # [2 * x + 7 * y,
        #  3 * y ** 2 + 7 * x]
        # Hessian function:
        # [[2, 7    ]
        #  [7, 6 * y]]
        hessian, gradient = find_hessian_and_gradient(f, [x, y])

        func = theano.function([x, y], [hessian, gradient])
        hessian_output, gradient_output = func(1, 2)

        np.testing.assert_array_equal(
            gradient_output,
            np.array([16, 19])
        )
        np.testing.assert_array_equal(
            hessian_output,
            np.array([
                [2, 7],
                [7, 12],
            ])
        )

    def test_hessian_assign_step_exception(self):
        with self.assertRaises(ValueError):
            algorithms.Hessian((2, 3, 1), step=0.01)
