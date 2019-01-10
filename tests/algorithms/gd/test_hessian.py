from functools import partial

import numpy as np
import tensorflow as tf

from neupy import algorithms, layers
from neupy.utils import tensorflow_session
from neupy.algorithms.gd.hessian import find_hessian_and_gradient

from helpers import compare_networks
from helpers import simple_classification
from base import BaseTestCase


class HessianTestCase(BaseTestCase):
    def test_compare_bp_and_hessian(self):
        x_train, x_test, y_train, y_test = simple_classification()
        compare_networks(
            # Test classes
            partial(algorithms.GradientDescent, batch_size=None),
            partial(algorithms.Hessian, penalty_const=1),
            # Test data
            (x_train, y_train, x_test, y_test),
            # Network configurations
            network=[
                layers.Input(10),
                layers.Sigmoid(15),
                layers.Sigmoid(1)
            ],
            shuffle_data=True,
            verbose=False,
            show_epoch=1,
            # Test configurations
            epochs=5,
            show_comparison_plot=False
        )

    def test_hessian_computation(self):
        x = tf.placeholder(name='x', dtype=tf.float32, shape=(1,))
        y = tf.placeholder(name='y', dtype=tf.float32, shape=(1,))

        f = x ** 2 + y ** 3 + 7 * x * y
        # Gradient function:
        # [2 * x + 7 * y,
        #  3 * y ** 2 + 7 * x]
        # Hessian function:
        # [[2, 7    ]
        #  [7, 6 * y]]
        hessian, gradient = find_hessian_and_gradient(f, [x, y])

        session = tensorflow_session()
        hessian_output, gradient_output = session.run(
            [hessian, gradient], feed_dict={x: [1], y: [2]})

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
            # Don't have step parameter
            algorithms.Hessian(
                layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1),
                step=0.01,
            )

    def test_hessian_overfit(self):
        self.assertCanNetworkOverfit(
            partial(algorithms.Hessian, verbose=False, penalty_const=0.1),
            epochs=350,
        )
