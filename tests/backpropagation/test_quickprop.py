from functools import partial

import numpy as np
from sklearn import datasets, cross_validation, preprocessing

from neupy import algorithms, layers

from utils import compare_networks
from base import BaseTestCase


class QuickPropTestCase(BaseTestCase):
    def setUp(self):
        super(QuickPropTestCase, self).setUp()
        data, target = datasets.make_regression(n_samples=1500, n_features=5,
                                                n_informative=5, n_targets=1,
                                                random_state=33)
        target_scaler = preprocessing.MinMaxScaler()
        target = target_scaler.fit_transform(target.reshape(-1, 1))
        self.data = cross_validation.train_test_split(data, target,
                                                      train_size=0.75)
        self.connection = (5, 10, 1)

    def test_quickprop(self):
        x_train, x_test, y_train, y_test = self.data

        qp = algorithms.Quickprop(
            (5, 10, 1),
            step=0.1,
            upper_bound=1,
            use_raw_predict_at_error=False,
            shuffle_data=True,
            # verbose=True,
        )
        qp.train(x_train, y_train, epochs=100)

        result = qp.predict(x_test)
        error = qp.error(result, y_test)
        self.assertAlmostEqual(0, error, places=2)

    def test_compare_quickprop_and_bp(self):
        x_train, _, y_train, _ = self.data
        network_default_error, network_tested_error = compare_networks(
            # Test classes
            algorithms.Backpropagation,
            partial(algorithms.Quickprop, upper_bound=0.5),
            # Test data
            (x_train, y_train),
            # Network configurations
            connection=self.connection,
            step=0.1,
            use_raw_predict_at_error=False,
            shuffle_data=True,
            # Test configurations
            epochs=100,
            verbose=False,
            is_comparison_plot=False
        )
        self.assertGreater(network_default_error, network_tested_error)
