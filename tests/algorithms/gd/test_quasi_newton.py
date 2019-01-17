from functools import partial
from collections import namedtuple

import numpy as np
from sklearn import metrics

from neupy.utils import asfloat
from neupy import algorithms, layers, init
from neupy.algorithms.gd import quasi_newton as qn

from data import simple_classification
from base import BaseTestCase


class QuasiNewtonTestCase(BaseTestCase):
    def test_exceptions(self):
        with self.assertRaises(ValueError):
            # Don't have learning rate
            algorithms.QuasiNewton((2, 3, 1), step=0.3)

    def test_update_functions(self):
        UpdateFunction = namedtuple(
            "UpdateFunction",
            "func input_values output_value is_symmetric")

        testcases = (
            UpdateFunction(
                func=qn.bfgs,
                input_values=[
                    asfloat(np.eye(3)),
                    asfloat(np.array([0.1, 0.2, 0.3])),
                    asfloat(np.array([0.3, -0.3, -0.5]))
                ],
                output_value=np.array([
                    [1.41049383, 0.32098765, 0.4537037],
                    [0.32098765, 0.64197531, -0.59259259],
                    [0.4537037, -0.59259259, 0.02777778],
                ]),
                is_symmetric=True
            ),
            UpdateFunction(
                func=qn.sr1,
                input_values=[
                    asfloat(np.eye(3)),
                    asfloat(np.array([0.1, 0.2, 0.3])),
                    asfloat(np.array([1, -2, -3]))
                ],
                output_value=[
                    [0.94671053, 0.13026316, 0.19539474],
                    [0.13026316, 0.68157895, -0.47763158],
                    [0.19539474, -0.47763158, 0.28355263],
                ],
                is_symmetric=False
            ),
            UpdateFunction(
                func=qn.dfp,
                input_values=[
                    asfloat(np.eye(3)),
                    asfloat(np.array([0.1, 0.2, 0.3])),
                    asfloat(np.array([0.3, -0.3, -0.5]))
                ],
                output_value=np.array([
                    [0.73514212, 0.09819121, 0.18217053],
                    [0.09819121, 0.56847545, -0.6821705],
                    [0.18217053, -0.6821705, -0.08139535],
                ]),
                is_symmetric=True
            ),
        )

        for case in testcases:
            if not case.input_values:
                continue

            result = self.eval(case.func(*case.input_values))

            if case.is_symmetric:
                np.testing.assert_array_almost_equal(result, result.T)

            np.testing.assert_array_almost_equal(result, case.output_value)

    def test_quasi_newton_bfgs(self):
        x_train, x_test, y_train, y_test = simple_classification()

        qnnet = algorithms.QuasiNewton(
            connection=[
                layers.Input(10),
                layers.Sigmoid(30, weight=init.Orthogonal()),
                layers.Sigmoid(1, weight=init.Orthogonal()),
            ],
            shuffle_data=True,
            show_epoch='20 times',
            update_function='bfgs',
        )

        qnnet.train(x_train, y_train, x_test, y_test, epochs=50)
        result = qnnet.predict(x_test).round().astype(int)

        roc_curve_score = metrics.roc_auc_score(result, y_test)
        self.assertAlmostEqual(0.92, roc_curve_score, places=2)

    def test_quasi_newton_dfp(self):
        x_train, x_test, y_train, y_test = simple_classification()

        qnnet = algorithms.QuasiNewton(
            connection=[
                layers.Input(10),
                layers.Sigmoid(30, weight=init.Orthogonal()),
                layers.Sigmoid(1, weight=init.Orthogonal()),
            ],
            shuffle_data=True,
            verbose=False,

            update_function='dfp',
            h0_scale=2,
        )
        qnnet.train(x_train, y_train, x_test, y_test, epochs=10)
        result = qnnet.predict(x_test).round()

        roc_curve_score = metrics.roc_auc_score(result, y_test)
        self.assertAlmostEqual(0.92, roc_curve_score, places=2)

    def test_quasi_newton_sr1(self):
        x_train, x_test, y_train, y_test = simple_classification()

        qnnet = algorithms.QuasiNewton(
            connection=[
                layers.Input(10),
                layers.Sigmoid(30, weight=init.Orthogonal()),
                layers.Sigmoid(1, weight=init.Orthogonal()),
            ],
            shuffle_data=True,
            show_epoch=20,
            verbose=False,

            update_function='sr1',
            h0_scale=2,
        )
        qnnet.train(x_train, y_train, x_test, y_test, epochs=10)
        result = qnnet.predict(x_test).round()

        roc_curve_score = metrics.roc_auc_score(result, y_test)
        self.assertAlmostEqual(0.92, roc_curve_score, places=2)

    def test_quasi_newton_bfgs_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.QuasiNewton,
                h0_scale=2,
                update_function='bfgs',
                verbose=False,
            ),
            epochs=100,
            min_accepted_error=0.002,
        )

    def test_quasi_newton_dfp_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.QuasiNewton,
                update_function='dfp',
                verbose=False,
            ),
            epochs=350,
            min_accepted_error=0.002,
        )

    def test_quasi_newton_sr1_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.QuasiNewton,
                update_function='sr1',
                verbose=False,
                epsilon=1e-5,
            ),
            epochs=250,
            min_accepted_error=0.002,
        )

    def test_safe_division(self):
        value = self.eval(qn.safe_division(2.0, 4.0, epsilon=1e-7))
        self.assertAlmostEqual(0.5, value)

        value = self.eval(qn.safe_division(1.0, 1e-8, epsilon=1e-7))
        self.assertAlmostEqual(1e7, value)

    def test_safe_reciprocal(self):
        value = self.eval(qn.safe_reciprocal(4.0, epsilon=1e-7))
        self.assertAlmostEqual(0.25, value)

        value = self.eval(qn.safe_reciprocal(1e-8, epsilon=1e-7))
        self.assertAlmostEqual(1e7, value)

        value = self.eval(qn.safe_reciprocal(1e-8, epsilon=0.01))
        self.assertAlmostEqual(100, value)
