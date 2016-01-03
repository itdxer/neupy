import unittest
from collections import namedtuple
from functools import partial

import theano
import numpy as np
from sklearn import datasets, metrics
from sklearn.cross_validation import StratifiedShuffleSplit

from neupy import algorithms, layers
from neupy.algorithms.gd.quasi_newton import line_search
from neupy.algorithms.gd import quasi_newton as qn

from data import simple_classification
from base import BaseTestCase


class QuasiNewtonTestCase(BaseTestCase):
    def test_line_search_exceptions(self):
        testcases = [
            # Invalid c1 values
            dict(c1=-1, c2=0.5, amax=1),
            dict(c1=0, c2=0.5, amax=1),
            dict(c1=1, c2=0.5, amax=1),

            # Invalid c2 values
            dict(c2=-1, c1=0.5, amax=1),
            dict(c2=0, c1=0.5, amax=1),
            dict(c2=1, c1=0.5, amax=1),

            # c1 > c2
            dict(c1=0.5, c2=0.1, amax=1),

            # Invalid amax values
            dict(c1=0.05, c2=0.1, amax=-10),
            dict(c1=0.05, c2=0.1, amax=0),
        ]

        for testcase in testcases:
            error_desc = "Line search for {}".format(testcase)
            with self.assertRaises(ValueError, msg=error_desc):
                line_search(**testcase)

    def test_bfgs(self):
        x_train, x_test, y_train, y_test = simple_classification()

        qnnet = algorithms.QuasiNewton(
            connection=[
                layers.Sigmoid(10, init_method='ortho'),
                layers.Sigmoid(20, init_method='ortho'),
                layers.Output(1)
            ],
            step=0.1,
            shuffle_data=True,
            show_epoch='20 times',
            verbose=False,
        )
        qnnet.train(x_train, y_train, x_test, y_test, epochs=20)
        result = qnnet.predict(x_test).round()

        roc_curve_score = metrics.roc_auc_score(result, y_test)
        self.assertAlmostEqual(0.92, roc_curve_score, places=2)

    def test_update_functions(self):
        UpdateFunction = namedtuple("UpdateFunction",
                                    "func input_values output_value "
                                    "is_symmetric")
        testcases = (
            UpdateFunction(
                func=qn.bfgs,
                input_values=[
                    np.eye(3),
                    np.array([0.1, 0.2, 0.3]),
                    np.array([0.3, -0.3, -0.5])
                ],
                output_value=np.array([
                    [1.41049383, 0.32098765, 0.4537037],
                    [0.32098765, 0.64197531, -0.59259259],
                    [0.4537037, -0.59259259, 0.02777778]
                ]),
                is_symmetric=True
            ),
            UpdateFunction(
                func=qn.sr1,
                input_values=[
                    np.eye(3),
                    np.array([0.1, 0.2, 0.3]),
                    np.array([1, -2, -3])
                ],
                output_value=[
                    [0.94671053, 0.13026316, 0.19539474],
                    [0.13026316, 0.68157895, -0.47763158],
                    [0.19539474, -0.47763158, 0.28355263],
                ],
                is_symmetric=False
            ),
            UpdateFunction(
                func=qn.psb,
                input_values=[
                    np.eye(3),
                    np.array([0.12, 0.29, 0.33]),
                    np.array([1.1, -2.01, -3.73])
                ],
                output_value=[
                    [0.95617560, 0.10931254, 0.19090481],
                    [0.10931254, 0.74683874, -0.44796302],
                    [0.19090481, -0.44796302, 0.20922278],
                ],
                is_symmetric=True
            ),
            UpdateFunction(
                func=qn.dfp,
                input_values=[
                    np.eye(3),
                    np.array([0.1, 0.2, 0.3]),
                    np.array([0.3, -0.3, -0.5])
                ],
                output_value=np.array([
                    [0.73514212, -0.11111111, -0.16666667],
                    [-0.11111111, 0.56847545, -0.33333333],
                    [-0.16666667, -0.33333333, -0.08139535],
                ]),
                is_symmetric=True
            ),
        )

        for case in testcases:
            if not case.input_values:
                continue

            values = map(theano.shared, case.input_values)
            result = case.func(*values).eval()

            if case.is_symmetric:
                np.testing.assert_array_almost_equal(result, result.T)

            np.testing.assert_array_almost_equal(result, case.output_value)

    def test_quasi_newton_dfp(self):
        x_train, x_test, y_train, y_test = simple_classification()

        qnnet = algorithms.QuasiNewton(
            connection=[
                layers.Sigmoid(10, init_method='ortho'),
                layers.Sigmoid(30, init_method='ortho'),
                layers.Output(1)
            ],
            step=0.1,
            shuffle_data=True,
            show_epoch=20,
            verbose=False,

            update_function='dfp',
            h0_scale=2,
            gradient_tol=1e-5,
        )
        qnnet.train(x_train, y_train, x_test, y_test, epochs=10)
        result = qnnet.predict(x_test).round()

        roc_curve_score = metrics.roc_auc_score(result, y_test)
        self.assertAlmostEqual(0.92, roc_curve_score, places=2)

    def test_quasi_newton_psb(self):
        x_train, x_test, y_train, y_test = simple_classification()

        qnnet = algorithms.QuasiNewton(
            connection=[
                layers.Sigmoid(10, init_method='ortho'),
                layers.Sigmoid(30, init_method='ortho'),
                layers.Output(1)
            ],
            step=0.1,
            shuffle_data=True,
            show_epoch=20,
            verbose=False,

            update_function='psb',
            h0_scale=2,
            gradient_tol=1e-10,
        )
        qnnet.train(x_train, y_train, x_test, y_test, epochs=10)
        result = qnnet.predict(x_test).round()

        roc_curve_score = metrics.roc_auc_score(result, y_test)
        self.assertAlmostEqual(0.92, roc_curve_score, places=2)

    def test_quasi_newton_sr1(self):
        x_train, x_test, y_train, y_test = simple_classification()

        qnnet = algorithms.QuasiNewton(
            connection=[
                layers.Sigmoid(10, init_method='ortho'),
                layers.Sigmoid(30, init_method='ortho'),
                layers.Output(1)
            ],
            step=0.1,
            shuffle_data=True,
            show_epoch=20,
            verbose=False,

            update_function='sr1',
            h0_scale=2,
            gradient_tol=1e-10,
        )
        qnnet.train(x_train, y_train, x_test, y_test, epochs=10)
        result = qnnet.predict(x_test).round()

        roc_curve_score = metrics.roc_auc_score(result, y_test)
        self.assertAlmostEqual(0.92, roc_curve_score, places=2)
