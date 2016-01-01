import unittest
from collections import namedtuple
from functools import partial

import numpy as np

from sklearn import datasets, metrics
from sklearn.cross_validation import StratifiedShuffleSplit
from neupy import algorithms, layers

from data import simple_classification
from base import BaseTestCase


class BFGSTestCase(BaseTestCase):
    def test_bfgs(self):
        x_train, x_test, y_train, y_test = simple_classification()

        qnnet = algorithms.BFGS(
            connection=[
                layers.Sigmoid(10, init_method='ortho'),
                layers.Sigmoid(20, init_method='ortho'),
                layers.Output(1)
            ],
            step=0.1,
            shuffle_data=True,
            show_epoch='20 times',
            verbose=True,
        )
        qnnet.train(x_train, y_train, x_test, y_test, epochs=1000)
        result = qnnet.predict(x_test).round()

        roc_curve_score = metrics.roc_auc_score(result, y_test)
        self.assertAlmostEqual(0.92, roc_curve_score, places=2)
