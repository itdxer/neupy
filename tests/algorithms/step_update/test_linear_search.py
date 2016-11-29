import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split

from neupy import algorithms, layers
from neupy.estimators import rmsle

from base import BaseTestCase


class LinearSearchTestCase(BaseTestCase):
    def test_linear_search(self):
        methods = [
            ('golden', 0.34276),
            ('brent', 0.35192),
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
                    layers.Input(13),
                    layers.Sigmoid(50),
                    layers.Sigmoid(1),
                ],
                show_epoch=1,
                verbose=False,
                search_method=method_name,
                tol=0.1,
                addons=[algorithms.LinearSearch],
            )
            cgnet.train(x_train, y_train, epochs=4)
            y_predict = cgnet.predict(x_test).round(1)

            error = rmsle(target_scaler.inverse_transform(y_test),
                          target_scaler.inverse_transform(y_predict))

            self.assertAlmostEqual(valid_error, error, places=5)
