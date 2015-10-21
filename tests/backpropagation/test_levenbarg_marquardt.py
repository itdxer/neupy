import numpy as np

from sklearn import datasets, preprocessing
from sklearn.cross_validation import train_test_split
from neupy import algorithms, layers
from neupy.functions import rmsle

from base import BaseTestCase


class LevenbergMarquardtTestCase(BaseTestCase):
    def test_levenberg_marquardt(self):
        dataset = datasets.load_diabetes()
        data, target = dataset.data, dataset.target

        data_scaler = preprocessing.MinMaxScaler()
        target_scaler = preprocessing.MinMaxScaler()

        x_train, x_test, y_train, y_test = train_test_split(
            data_scaler.fit_transform(data),
            target_scaler.fit_transform(target.reshape(-1, 1)),
            train_size=0.85
        )

        # Network
        lmnet = algorithms.LevenbergMarquardt(
            connection=[
                layers.SigmoidLayer(10),
                layers.SigmoidLayer(40),
                layers.OutputLayer(1),
            ],
            mu_increase_factor=2,
            mu=0.1,
            show_epoch=10,
            use_bias=False,
            verbose=False,
        )
        lmnet.train(x_train, y_train, epochs=100)
        y_predict = lmnet.predict(x_test)

        error = rmsle(target_scaler.inverse_transform(y_test),
                      target_scaler.inverse_transform(y_predict).round())
        error

        self.assertAlmostEqual(0.4372, error, places=4)
