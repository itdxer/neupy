from functools import partial

import tensorflow as tf

from neupy import algorithms

from data import simple_classification
from base import BaseTestCase


class BackPropAlgsTestCase(BaseTestCase):
    def setUp(self):
        super(BackPropAlgsTestCase, self).setUp()
        self.bp_algorithms = [
            partial(algorithms.GradientDescent, batch_size='all'),
            algorithms.GradientDescent,
            algorithms.ConjugateGradient,
            algorithms.HessianDiagonal,
            algorithms.Hessian,
            algorithms.LevenbergMarquardt,
            algorithms.Momentum,
            algorithms.QuasiNewton,
            algorithms.RPROP,
            algorithms.IRPROPPlus,
            algorithms.Adadelta,
            algorithms.Adagrad,
            algorithms.Adam,
            algorithms.Adamax,
            algorithms.RMSProp,
        ]

    def test_custom_error_functions(self):
        # Test that everything works without fail
        def custom_mse(expected, predicted):
            return tf.reduce_mean(0.5 * (predicted - expected) ** 2)

        x_train, _, y_train, _ = simple_classification()
        gdnet = algorithms.GradientDescent(
            (10, 10, 1), error=custom_mse, batch_size='all')

        gdnet.train(x_train, y_train)
