from sklearn import datasets, model_selection, metrics
from neupy import algorithms, init
from neupy.layers import Relu, Sigmoid, Input

from base import BaseTestCase


class DANTestCase(BaseTestCase):
    def test_handle_errors(self):
        data, target = datasets.make_classification(300, n_features=4,
                                                    n_classes=2)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            data, target, train_size=0.7
        )

        with self.assertRaises(ValueError):
            # First network has two output layers and the second
            # just one.
            algorithms.DynamicallyAveragedNetwork([
                algorithms.RPROP((4, 10, 2), step=0.1),
                algorithms.GradientDescent((4, 10, 1), step=0.1)
            ])

        with self.assertRaises(ValueError):
            # Use ensemble with less than one network
            algorithms.DynamicallyAveragedNetwork([
                algorithms.GradientDescent((4, 10, 1), step=0.1)
            ])

        with self.assertRaises(ValueError):
            # Output greater than 1
            dan = algorithms.DynamicallyAveragedNetwork([
                algorithms.GradientDescent(
                    [
                        Input(4),
                        Sigmoid(10),
                        Relu(1, weight=init.Uniform(), bias=init.Uniform()),
                    ],
                    step=0.01
                ),
                algorithms.RPROP((4, 10, 1), step=0.01),
            ])
            dan.train(x_train, y_train, epochs=10)
            dan.predict(x_test)

    def test_dan(self):
        data, target = datasets.make_classification(300, n_features=4,
                                                    n_classes=2)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            data, target, train_size=0.7
        )

        dan = algorithms.DynamicallyAveragedNetwork([
            algorithms.RPROP((4, 10, 1), step=0.1, maxstep=1),
            algorithms.GradientDescent((4, 5, 1), step=0.1),
            algorithms.ConjugateGradient((4, 5, 1), step=0.01),
        ])

        dan.train(x_train, y_train, epochs=500)
        result = dan.predict(x_test)
        ensemble_result = metrics.accuracy_score(y_test, result)
        self.assertAlmostEqual(0.9444, ensemble_result, places=4)

    def test_dan_repr(self):
        dan = algorithms.DynamicallyAveragedNetwork([
            algorithms.Momentum((3, 2, 1)),
            algorithms.GradientDescent((3, 2, 1)),
        ])
        dan_repr = str(dan)

        self.assertIn('DynamicallyAveragedNetwork', dan_repr)
        self.assertIn('Momentum', dan_repr)
        self.assertIn('GradientDescent', dan_repr)
