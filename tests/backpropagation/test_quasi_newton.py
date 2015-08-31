import unittest
from collections import namedtuple
from functools import partial

import numpy as np

from sklearn import datasets, metrics
from sklearn.cross_validation import StratifiedShuffleSplit
from neuralpy import algorithms, layers
from neuralpy.algorithms.backprop.quasi_newton import bfgs, dfp, sr1, psb

from data import simple_input_train, simple_target_train
from utils import compare_networks
from base import BaseTestCase


class QuasiNewtonTestCase(BaseTestCase):
    def setUp(self):
        super(QuasiNewtonTestCase, self).setUp()

        X, y = datasets.make_classification(n_samples=100, n_features=10,
                                            random_state=33)
        shuffle_split = StratifiedShuffleSplit(y, 1, train_size=0.6,
                                               random_state=33)

        train_index, test_index = next(shuffle_split.__iter__())
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        self.X, self.y = X, y
        self.data = (x_train, x_test, y_train, y_test)

    def test_quasi_newton_bfgs(self):
        x_train, x_test, y_train, y_test = self.data

        qnnet = algorithms.QuasiNewton(
            connection=[
                layers.SigmoidLayer(10, init_method='ortho'),
                layers.SigmoidLayer(20, init_method='ortho'),
                layers.OutputLayer(1)
            ],
            step=0.1,
            use_raw_predict_at_error=False,
            shuffle_data=True,
            show_epoch=20,
            verbose=False,

            update_function='bfgs',
            h0_scale=5,
            gradient_tol=1e-5,
        )
        qnnet.train(x_train, y_train, x_test, y_test, epochs=100)
        result = qnnet.predict(x_test).round()

        roc_curve_score = metrics.roc_auc_score(result, y_test)
        self.assertAlmostEqual(0.92, roc_curve_score, places=2)

    def test_quasi_newton_dfp(self):
        x_train, x_test, y_train, y_test = self.data

        qnnet = algorithms.QuasiNewton(
            connection=[
                layers.SigmoidLayer(10, init_method='ortho'),
                layers.SigmoidLayer(30, init_method='ortho'),
                layers.OutputLayer(1)
            ],
            step=0.1,
            use_raw_predict_at_error=False,
            shuffle_data=True,
            show_epoch=20,
            verbose=False,

            update_function='bfgs',
            h0_scale=2,
            gradient_tol=1e-5,

            maxstep=20,
        )
        qnnet.train(x_train, y_train, x_test, y_test, epochs=10)
        result = qnnet.predict(x_test).round()

        roc_curve_score = metrics.roc_auc_score(result, y_test)
        self.assertAlmostEqual(0.92, roc_curve_score, places=2)

    def test_default_optimization(self):
        qnnet = algorithms.QuasiNewton((2, 3, 1))
        self.assertEqual(qnnet.optimizations,
                         [algorithms.WolfeSearch])

        qnnet = algorithms.QuasiNewton((2, 3, 1),
                                       optimizations=[algorithms.WeightDecay])
        self.assertEqual(qnnet.optimizations,
                         [algorithms.WeightDecay, algorithms.WolfeSearch])

        qnnet = algorithms.QuasiNewton((2, 3, 1),
                                       optimizations=[algorithms.WeightDecay,
                                                      algorithms.LinearSearch])
        self.assertEqual(qnnet.optimizations,
                         [algorithms.WeightDecay, algorithms.LinearSearch])

    def test_update_functions(self):
        UpdateFunction = namedtuple("UpdateFunction",
                                    "func input_values output_value "
                                    "is_symmetric")
        testcases = (
            UpdateFunction(
                func=bfgs,
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
                func=sr1,
                input_values=[],
                output_value=[],
                is_symmetric=True
            ),
            UpdateFunction(
                func=psb,
                input_values=[],
                output_value=[],
                is_symmetric=True
            ),
            UpdateFunction(
                func=dfp,
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

            result = case.func(*case.input_values)

            if case.is_symmetric:
                self.assertTrue(np.allclose(result, result.T))

            self.assertTrue(np.allclose(result, case.output_value))
