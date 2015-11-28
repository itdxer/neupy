from functools import partial

import numpy as np
from neupy.layers import Tanh, Sigmoid, StepOutput, Output
from neupy import algorithms
from sklearn import datasets, preprocessing
from sklearn.cross_validation import train_test_split

from data import simple_classification
from utils import compare_networks
from base import BaseTestCase


class GradientDescentTestCase(BaseTestCase):
    def test_stochastic_gradient_descent(self):
        output = StepOutput(1, output_bounds=(-1, 1))
        connection = Tanh(10) > Tanh(20) > output

        x_train, _, y_train, _ = simple_classification()

        compare_networks(
           # Test classes
           algorithms.GradientDescent,
           partial(algorithms.MinibatchGradientDescent, batch_size=1),
           # Test data
           (x_train, y_train),
           # Network configurations
           connection=connection,
           step=0.1,
           shuffle_data=True,
           verbose=True,
           # Test configurations
           epochs=40,
           show_comparison_plot=True
        )

    def test_on_bigger_dataset(self):
        data, targets = datasets.make_regression(n_samples=400,
                                                 n_features=10)
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        targets = scaler.fit_transform(targets.reshape(-1, 1))
        x_train, x_test, y_train, y_test = train_test_split(
            data, targets, train_size=0.7
        )

        in_size = data.shape[1]
        out_size = targets.shape[1] if len(targets.shape) > 1 else 1

        sgd_network = algorithms.MinibatchGradientDescent(
            Sigmoid(in_size) > Sigmoid(300) > Output(out_size),
            step=0.2,
            batch_size=5,
            verbose=False,
        )
        sgd_network.train(x_train, y_train, x_test, y_test, epochs=20)
        test_error = sgd_network.prediction_error(x_test, y_test)
        self.assertAlmostEqual(0.007, test_error, places=3)
