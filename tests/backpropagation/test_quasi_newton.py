import numpy as np

from sklearn import datasets, metrics
from sklearn.cross_validation import StratifiedShuffleSplit
from neuralpy import algorithms, layers

from data import simple_input_train, simple_target_train
from base import BaseTestCase


class QuasiNewtonTestCase(BaseTestCase):
    def test_quasi_newton(self):
        X, y = datasets.make_classification(n_samples=1000, n_features=10)
        shuffle_split = StratifiedShuffleSplit(y, 1, train_size=0.6)
        # import pdb;pdb.set_trace()
        train_index, test_index = next(shuffle_split.__iter__())
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        qnnet = algorithms.QuasiNewton(
            connection=[
                layers.SigmoidLayer(10),
                layers.SigmoidLayer(100),
                layers.OutputLayer(1)
            ],
            step=0.3,
            use_raw_predict_at_error=False,
            shuffle_data=True,
            update_function='dfp',
        )
        qnnet.train(x_train, y_train, epochs=100)
        result = qnnet.predict(x_test).round()
        roc_curve_score = metrics.roc_auc_score(result, y_test)
        self.assertAlmostEqual(0.92, roc_curve_score, places=2)
