import warnings

import mock
import pytest
import matplotlib.pyplot as plt

from neupy import algorithms, layers
from neupy.algorithms.plots import load_pandas_module

from helpers import simple_classification
from base import BaseTestCase


class PlotsTestCase(BaseTestCase):
    def setUp(self):
        super(PlotsTestCase, self).setUp()
        self.network = layers.join(
            layers.Input(10),
            layers.Sigmoid(20),
            layers.Sigmoid(1),
        )

    def test_failed_pandas_import(self):
        with mock.patch('pkgutil.find_loader') as mock_find_loader:
            mock_find_loader.return_value = None

            message = "The `pandas` library is not installed"
            with self.assertRaisesRegexp(ImportError, message):
                load_pandas_module()

    def test_not_tarining_data_to_plot(self):
        optimizer = algorithms.Adadelta(self.network)

        with warnings.catch_warnings(record=True) as warns:
            optimizer.plot_errors()
            self.assertEqual(1, len(warns))
            self.assertEqual(
                str(warns[-1].message),
                "There is no data to plot")

    @pytest.mark.mpl_image_compare
    def test_plot_errors_no_batch(self):
        x_train, x_test, y_train, y_test = simple_classification()

        optimizer = algorithms.Adadelta(self.network, batch_size=None)
        optimizer.train(x_train, y_train, x_test, y_test, epochs=10)
        optimizer.plot_errors(show=False)

        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_plot_errors_batch(self):
        x_train, x_test, y_train, y_test = simple_classification()

        optimizer = algorithms.Adadelta(
            self.network,
            shuffle_data=True,
            batch_size=10,
        )
        optimizer.train(x_train, y_train, x_test, y_test, epochs=100)
        optimizer.plot_errors(show=False)

        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_plot_errors_show_triggered_automatically(self):
        x_train, x_test, y_train, y_test = simple_classification()

        optimizer = algorithms.Adadelta(
            self.network,
            shuffle_data=True,
            batch_size=10,
        )
        optimizer.train(x_train, y_train, epochs=100)
        events = []

        def mocked_show(*args, **kwargs):
            events.append('show')

        with mock.patch('matplotlib.pyplot.show', side_effect=mocked_show):
            optimizer.plot_errors(show=True)
            self.assertSequenceEqual(events, ['show'])

        return plt.gcf()
