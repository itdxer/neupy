import numpy as np

from sklearn import datasets, preprocessing, cross_validation, metrics
from neupy import algorithms, ensemble, layers
from neupy.layers import TanhLayer, SigmoidLayer, OutputLayer

from base import BaseTestCase


class DANTestCase(BaseTestCase):
    def test_handle_errors(self):
        data, target = datasets.make_classification(300, n_features=4,
                                                    n_classes=2)
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
            data, target, train_size=0.7
        )

        with self.assertRaises(ValueError):
            # First network has two output layers and the second
            # just one.
            ensemble.DynamicallyAveragedNetwork([
                algorithms.RPROP((4, 10, 2), step=0.1),
                algorithms.Backpropagation((4, 10, 1), step=0.1)
            ])

        with self.assertRaises(ValueError):
            # Use ensemble with less than one network
            ensemble.DynamicallyAveragedNetwork([
                algorithms.Backpropagation((4, 10, 1), step=0.1)
            ])

        with self.assertRaises(ValueError):
            # Output between -1 and 1
            dan = ensemble.DynamicallyAveragedNetwork([
                algorithms.Backpropagation(
                    SigmoidLayer(4) > TanhLayer(10) > OutputLayer(1),
                    step=0.01
                ),
                algorithms.RPROP((4, 10, 1), step=0.1)
            ])
            dan.train(x_train, y_train, epochs=10)
            dan.predict(x_test)

    def test_dan(self):
        data, target = datasets.make_classification(300, n_features=4,
                                                    n_classes=2)
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
            data, target, train_size=0.7
        )

        dan = ensemble.DynamicallyAveragedNetwork([
            algorithms.RPROP((4, 100, 1), step=0.1, maximum_step=1),
            algorithms.Backpropagation((4, 5, 1), step=0.1),
            algorithms.ConjugateGradient((4, 5, 1), step=0.01),
        ])

        dan.train(x_train, y_train, epochs=500)
        result = dan.predict(x_test)
        ensemble_result = metrics.accuracy_score(y_test, result)
        self.assertAlmostEqual(0.9333, ensemble_result, places=4)
