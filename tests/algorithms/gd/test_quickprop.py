from functools import partial

from sklearn import datasets, model_selection, preprocessing

from neupy import algorithms

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
        self.data = model_selection.train_test_split(data, target,
                                                     train_size=0.75)
        self.connection = (5, 10, 1)

    def test_quickprop(self):
        x_train, x_test, y_train, y_test = self.data

        qp = algorithms.Quickprop(
            (5, 10, 1),
            step=0.1,
            upper_bound=1,
            shuffle_data=True,
            verbose=False,
        )
        qp.train(x_train, y_train, epochs=50)

        error = qp.prediction_error(x_test, y_test)
        self.assertAlmostEqual(0, error, places=2)

    def test_compare_quickprop_and_bp(self):
        x_train, _, y_train, _ = self.data
        compare_networks(
            # Test classes
            algorithms.GradientDescent,
            partial(algorithms.Quickprop, upper_bound=0.5),
            # Test data
            (x_train, y_train),
            # Network configurations
            connection=self.connection,
            step=0.1,
            shuffle_data=True,
            # Test configurations
            epochs=100,
            verbose=False,
            show_comparison_plot=False
        )
