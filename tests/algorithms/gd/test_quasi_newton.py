import unittest
from collections import namedtuple
from functools import partial

import numpy as np

from sklearn import datasets, metrics
from sklearn.cross_validation import StratifiedShuffleSplit
from neupy import algorithms, layers
from neupy.algorithms.gd.quasi_newton import line_search

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
