import unittest
from collections import namedtuple

import numpy as np

from sklearn import datasets, metrics
from sklearn.cross_validation import StratifiedShuffleSplit
from neuralpy import algorithms, layers
from neuralpy.algorithms.backprop.quasi_newton import bfgs, dfp, sr1, psb

from data import simple_input_train, simple_target_train
from base import BaseTestCase


class QuasiNewtonTestCase(BaseTestCase):
    def test_quasi_newton(self):
        X, y = datasets.make_classification(n_samples=200, n_features=10,
                                            random_state=33)
        shuffle_split = StratifiedShuffleSplit(y, 1, train_size=0.6,
                                               random_state=33)

        train_index, test_index = next(shuffle_split.__iter__())
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        qnnet = algorithms.QuasiNewton(
            connection=[
                layers.SigmoidLayer(10),
                layers.SigmoidLayer(40),
                layers.OutputLayer(1)
            ],
            step=0.1,
            use_raw_predict_at_error=False,
            shuffle_data=True,
            update_function='sr1',
            show_epoch=20,
            search_method='brent',
            optimizations=[algorithms.LinearSearch]
        )
        qnnet.train(x_train, y_train, epochs=100)
        result = qnnet.predict(x_test).round()

        roc_curve_score = metrics.roc_auc_score(result, y_test)
        self.assertAlmostEqual(0.88, roc_curve_score, places=2)

    @unittest.skip("Not ready yet")
    def test_update_functions(self):
        UpdateFunction = namedtuple("UpdateFunction",
                                    "func input_values output_value")
        testcases = (
            UpdateFunction(
                func=bfgs,
                input_values=[],
                output_value=[]
            ),
            UpdateFunction(
                func=sr1,
                input_values=[],
                output_value=[]
            ),
            UpdateFunction(
                func=psb,
                input_values=[],
                output_value=[]
            ),
            UpdateFunction(
                func=dfp,
                input_values=[],
                output_value=[]
            ),
        )

        for case in testcases:
            result = case.func(*case.input_values)
            self.assertEqual(result, case.output_value)
