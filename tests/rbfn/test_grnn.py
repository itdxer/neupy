import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split

from neuralpy.algorithms import GRNN
from neuralpy.functions import rmsle
from base import BaseTestCase


class GRNNTestCase(BaseTestCase):
    def test_handle_errors(self):
        with self.assertRaises(ValueError):
            # Wrong: size of target data not the same as size of
            # input data.
            GRNN().train(
                np.array([[0], [0]]), np.array([0])
            )

        with self.assertRaises(ValueError):
            # Wrong: 2-D target vector (must be 1-D)
            GRNN().train(
                np.array([[0], [0]]), np.array([[0]])
            )

        with self.assertRaises(AttributeError):
            # Wrong: can't use iterative learning process for this
            # algorithm
            GRNN().train_epoch()

        with self.assertRaises(ValueError):
            # Wrong: invalid feature size for prediction data
            grnet = GRNN()
            grnet.train(np.array([[0], [0]]), np.array([0]))
            grnet.predict(np.array([[0]]))

    def test_simple_grnn(self):
        dataset = datasets.load_diabetes()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, train_size=0.7,
            random_state=0
        )

        nw = GRNN(standard_deviation=0.1)
        nw.train(x_train, y_train)
        result = nw.predict(x_test)
        error = rmsle(result, y_test)

        self.assertAlmostEqual(error, 0.4245, places=4)
