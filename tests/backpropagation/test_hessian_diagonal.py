import numpy as np

from sklearn import datasets, cross_validation, preprocessing
from neupy import algorithms, layers
from neupy.functions import rmsle

from data import simple_input_train, simple_target_train
from base import BaseTestCase


class QuasiNewtonTestCase(BaseTestCase):
    def test_hessian_diagonal(self):
        dataset = datasets.load_diabetes()
        data, target = dataset.data, dataset.target

        input_scaler = preprocessing.StandardScaler()
        target_scaler = preprocessing.StandardScaler()

        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
            input_scaler.fit_transform(data),
            target_scaler.fit_transform(target.reshape(-1, 1)),
            train_size=0.8
        )

        nw = algorithms.HessianDiagonal(
            connection=[
                layers.SigmoidLayer(10),
                layers.SigmoidLayer(20),
                layers.OutputLayer(1)
            ],
            step=1.5,
            use_raw_predict_at_error=False,
            shuffle_data=False,
            verbose=False,
            min_eigenvalue=1e-10
        )
        nw.train(x_train, y_train, epochs=10)
        y_predict = nw.predict(x_test)

        error = rmsle(target_scaler.inverse_transform(y_test),
                      target_scaler.inverse_transform(y_predict).round())

        self.assertAlmostEqual(0.5032, error, places=4)
