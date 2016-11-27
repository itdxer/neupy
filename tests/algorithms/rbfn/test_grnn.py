import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

from neupy import algorithms
from neupy.exceptions import NotTrained

from base import BaseTestCase


class GRNNTestCase(BaseTestCase):
    def test_grrn_exceptions(self):
        with self.assertRaises(ValueError):
            # size of target data not the same as size of
            # input data.
            grnet = algorithms.GRNN(verbose=False)
            grnet.train(np.array([[0], [0]]), np.array([0]))

        with self.assertRaises(ValueError):
            # 2 features for target data
            grnet = algorithms.GRNN(verbose=False)
            grnet.train(np.array([[0], [0]]), np.array([[0, 0]]))

        with self.assertRaises(ValueError):
            # invalid feature size for prediction data
            grnet = algorithms.GRNN(verbose=False)
            grnet.train(np.array([[0], [0]]), np.array([0]))
            grnet.predict(np.array([[0]]))

        with self.assertRaises(NotTrained):
            # Prediction without training
            grnet = algorithms.GRNN(verbose=False)
            grnet.predict(np.array([[0]]))

        with self.assertRaises(ValueError):
            # different number of features for
            # train and test data
            grnet = algorithms.GRNN(verbose=False)
            grnet.train(np.array([[0]]), np.array([0]))
            grnet.predict(np.array([[0, 0]]))

    def test_simple_grnn(self):
        dataset = datasets.load_diabetes()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, train_size=0.7
        )

        x_train_before = x_train.copy()
        x_test_before = x_test.copy()
        y_train_before = y_train.copy()

        grnnet = algorithms.GRNN(std=0.1, verbose=False)
        grnnet.train(x_train, y_train)
        result = grnnet.predict(x_test)
        error = metrics.mean_absolute_error(result, y_test)

        old_result = result.copy()
        self.assertAlmostEqual(error, 46.3358, places=4)

        # Test problem with variable links
        np.testing.assert_array_equal(x_train, x_train_before)
        np.testing.assert_array_equal(x_test, x_test_before)
        np.testing.assert_array_equal(y_train, y_train_before)

        x_train[:, :] = 0
        result = grnnet.predict(x_test)

        np.testing.assert_array_almost_equal(result, old_result)

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.GRNN(verbose=False),
            np.array([1, 2, 3]),
            np.array([1, 2, 3])
        )

    def test_predict_different_inputs(self):
        grnnet = algorithms.GRNN(verbose=False)

        data = np.array([[1, 2, 3]]).T
        target = np.array([[1, 2, 3]]).T

        grnnet.train(data, target)
        self.assertInvalidVectorPred(grnnet, data.ravel(), target,
                                     decimal=2)
