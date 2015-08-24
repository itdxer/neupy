from functools import partial

import numpy as np
from neuralpy.layers import (TanhLayer, SigmoidLayer, StepOutputLayer,
                             OutputLayer)
from neuralpy import algorithms
from sklearn import datasets, preprocessing
from sklearn.cross_validation import train_test_split

from data import xor_input_train, xor_target_train
from utils import compare_networks
from base import BaseTestCase


class GradientDescentTestCase(BaseTestCase):
    def setUp(self):
        super(GradientDescentTestCase, self).setUp()
        output = StepOutputLayer(1, output_bounds=(-1, 1))
        self.connection = TanhLayer(2) > TanhLayer(5) > output
    #
    def test_stochastic_gradient_descent(self):
        network_default_error, network_tested_error = compare_networks(
           # Test classes
           algorithms.Backpropagation,
           partial(algorithms.MinibatchGradientDescent, batch_size=4),
           # Test data
           (xor_input_train, xor_target_train),
           # Network configurations
           connection=self.connection,
           step=0.1,
           use_raw_predict_at_error=True,
           shuffle_data=True,
           # Test configurations
           epochs=40,
           # is_comparison_plot=True
        )
        self.assertGreater(network_default_error, network_tested_error)

    def test_on_bigger_dataset(self):
        data, targets = datasets.make_regression(n_samples=4000)
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        targets = scaler.fit_transform(targets)
        x_train, x_test, y_train, y_test = train_test_split(
            data, targets, train_size=0.7
        )

        in_size = data.shape[1]
        out_size = targets.shape[1] if len(targets.shape) > 1 else 1

        sgd_network = algorithms.MinibatchGradientDescent(
            SigmoidLayer(in_size) > SigmoidLayer(300) > OutputLayer(out_size),
            step=0.2,
            batch_size=10
        )
        sgd_network.train(x_train, y_train, x_test, y_test, epochs=10)
        result = sgd_network.predict(x_test)
        test_error = sgd_network.error(result,
                                       np.reshape(y_test, (y_test.size, 1)))
        self.assertAlmostEqual(0.02, test_error, places=2)
