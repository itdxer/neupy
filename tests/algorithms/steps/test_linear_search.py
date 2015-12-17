import numpy as np
import theano.tensor as T
from sklearn import datasets, preprocessing
from sklearn.cross_validation import train_test_split

from neupy import algorithms, layers
from neupy.algorithms.steps.linear_search import fmin_golden_search

from utils import rmsle
from base import BaseTestCase


class GoldenSearchTestCase(BaseTestCase):
    def test_golden_search_exceptions(self):
        invalid_parameters = (
            dict(tol=-1),
            dict(minstep=-1),
            dict(maxstep=-1),
            dict(maxiter=-1),
            dict(tol=0),
            dict(minstep=0),
            dict(maxstep=0),
            dict(maxiter=0),
        )
        for params in invalid_parameters:
            with self.assertRaises(ValueError):
                fmin_golden_search(lambda x: x, **params)

        with self.assertRaises(ValueError):
            fmin_golden_search(lambda x: x, minstep=10, maxstep=1)

    def test_golden_search_function(self):
        def f(x):
            return T.sin(x) * x ** -0.5

        def check_updates(step):
            return f(3 + step)

        best_step = fmin_golden_search(check_updates)
        self.assertAlmostEqual(1.6, best_step, places=2)

        best_step = fmin_golden_search(check_updates, maxstep=1)
        self.assertAlmostEqual(1, best_step, places=2)

    def test_linear_search(self):
        methods = [
            ('golden', 0.20976),
            # ('brent', 0.21190),
        ]

        for method_name, valid_error in methods:
            np.random.seed(self.random_seed)

            dataset = datasets.load_boston()
            data, target = dataset.data, dataset.target

            data_scaler = preprocessing.MinMaxScaler()
            target_scaler = preprocessing.MinMaxScaler()

            x_train, x_test, y_train, y_test = train_test_split(
                data_scaler.fit_transform(data),
                target_scaler.fit_transform(target.reshape(-1, 1)),
                train_size=0.85
            )

            cgnet = algorithms.ConjugateGradient(
                connection=[
                    layers.Sigmoid(13),
                    layers.Sigmoid(50),
                    layers.Output(1),
                ],
                search_method=method_name,
                show_epoch=25,
                optimizations=[algorithms.LinearSearch],
            )
            cgnet.train(x_train, y_train, epochs=100)
            y_predict = cgnet.predict(x_test).round(1)

            error = rmsle(target_scaler.inverse_transform(y_test),
                          target_scaler.inverse_transform(y_predict))

            self.assertAlmostEqual(valid_error, error, places=5)
