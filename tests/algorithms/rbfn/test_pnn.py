from __future__ import division

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold, train_test_split

from neupy import algorithms
from base import BaseTestCase


class PNNTestCase(BaseTestCase):
    def test_handle_errors(self):
        with self.assertRaises(ValueError):
            # Wrong: size of target data not the same as size of
            # input data.
            algorithms.PNN(verbose=False).train(
                np.array([[0], [0]]), np.array([0])
            )

        with self.assertRaises(ValueError):
            # Wrong: 2-D target vector (must be 1-D)
            algorithms.PNN(verbose=False).train(
                np.array([[0], [0]]), np.array([[0]])
            )

        with self.assertRaises(AttributeError):
            # Wrong: can't use iterative learning process for this
            # algorithm
            algorithms.PNN(verbose=False).train_epoch()

        with self.assertRaises(ValueError):
            # Wrong: invalid feature size for prediction data
            grnet = algorithms.PNN(verbose=False)
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

            nw = algorithms.PNN(verbose=False, std=0.1)
            nw.train(x_train, y_train)
            result = nw.predict(x_test)
            avarage_result += sum(y_test == result)

        self.assertEqual(avarage_result / test_data_size, 14.4)

    def test_digit_prediction(self):
        dataset = datasets.load_digits()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, train_size=0.7
        )

        nw = algorithms.PNN(verbose=False, std=10)
        nw.train(x_train, y_train)
        result = nw.predict(x_test)

        self.assertAlmostEqual(metrics.accuracy_score(y_test, result),
                               0.9889, places=4)

    def test_predict_probability(self):
        dataset = datasets.load_digits()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, train_size=0.7
        )

        x_train_before = x_train.copy()
        x_test_before = x_test.copy()
        y_train_before = y_train.copy()

        number_of_classes = len(np.unique(dataset.target))

        pnnet = algorithms.PNN(verbose=False, std=10)
        pnnet.train(x_train, y_train)
        result = pnnet.predict_proba(x_test)

        n_test_inputs = x_test.shape[0]
        self.assertEqual(result.shape, (n_test_inputs, number_of_classes))

        total_classes_prob = np.round(result.sum(axis=1), 10)
        np.testing.assert_array_equal(
            total_classes_prob,
            np.ones(n_test_inputs)
        )
        old_result = result.copy()

        # Test problem with variable links
        np.testing.assert_array_equal(x_train, x_train_before)
        np.testing.assert_array_equal(x_test, x_test_before)
        np.testing.assert_array_equal(y_train, y_train_before)

        x_train[:, :] = 0
        result = pnnet.predict_proba(x_test)
        total_classes_prob = np.round(result.sum(axis=1), 10)
        np.testing.assert_array_almost_equal(result, old_result)

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.PNN(verbose=False),
            np.array([1, 2, 3]),
            np.array([1, 0, 1])
        )

    def test_predict_different_inputs(self):
        pnnet = algorithms.PNN(verbose=False)

        data = np.array([[1, 2, 3]]).T
        target = np.array([[1, 0, 1]]).T

        pnnet.train(data, target)
        self.assertInvalidVectorPred(pnnet, data.ravel(), target.ravel(),
                                     decimal=2)
