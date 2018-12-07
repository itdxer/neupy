import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split

from neupy import algorithms, layers
from neupy.utils import asfloat
from neupy.algorithms.gd import errors

from base import BaseTestCase


class LinearSearchTestCase(BaseTestCase):
    single_thread = 1

    def test_linear_search(self):
        methods = [
            ('golden', 0.37381297),
            ('brent', 0.36021027),
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
                test_size=0.15
            )

            cgnet = algorithms.GradientDescent(
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
            cgnet.train(x_train, y_train, x_test, y_test, epochs=10)
            y_predict = cgnet.predict(x_test)

            error = errors.rmsle(
                asfloat(target_scaler.inverse_transform(y_test)),
                asfloat(target_scaler.inverse_transform(y_predict)),
            )
            error = self.eval(error)
            self.assertAlmostEqual(valid_error, error, places=5)
