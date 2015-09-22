from __future__ import division

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold, train_test_split

from neupy.algorithms import PNN
from base import BaseTestCase


class PNNTestCase(BaseTestCase):
    def test_handle_errors(self):
        with self.assertRaises(ValueError):
            # Wrong: size of target data not the same as size of
            # input data.
            PNN().train(np.array([[0], [0]]), np.array([0]))

        with self.assertRaises(ValueError):
            # Wrong: 2-D target vector (must be 1-D)
            PNN().train(np.array([[0], [0]]), np.array([[0]]))

        with self.assertRaises(AttributeError):
            # Wrong: can't use iterative learning process for this
            # algorithm
            PNN().train_epoch()

        with self.assertRaises(ValueError):
            # Wrong: invalid feature size for prediction data
            grnet = PNN()
            grnet.train(np.array([[0], [0]]), np.array([0]))
            grnet.predict(np.array([[0]]))

    def test_simple_pnn(self):
        dataset = datasets.load_iris()
        data = dataset.data
        target = dataset.target

        test_data_size = 10
        skfold = StratifiedKFold(target, test_data_size)
        avarage_result = 0

        for train, test in skfold:
            x_train, x_test = data[train], data[test]
            y_train, y_test = target[train], target[test]

            nw = PNN(standard_deviation=0.1)
            nw.train(x_train, y_train)
            result = nw.predict(x_test)
            avarage_result += sum(y_test == result)

        self.assertEqual(avarage_result / test_data_size, 14.4)

    def test_digit_prediction(self):
        dataset = datasets.load_digits()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, train_size=0.7
        )

        nw = PNN(standard_deviation=10)
        nw.train(x_train, y_train)
        result = nw.predict(x_test)

        self.assertAlmostEqual(metrics.accuracy_score(y_test, result),
                               0.9889, places=4)

    def test_predict_probability(self):
        dataset = datasets.load_digits()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, train_size=0.7
        )
        number_of_classes = len(np.unique(dataset.target))

        nw = PNN(standard_deviation=10)
        nw.train(x_train, y_train)
        result = nw.predict_prob(x_test)

        n_test_inputs = x_test.shape[0]
        self.assertEqual(result.shape, (n_test_inputs, number_of_classes))

        total_classes_prob = np.round(result.sum(axis=1), 10)
        self.assertTrue(
            np.all(total_classes_prob == np.ones((n_test_inputs, 1)))
        )
