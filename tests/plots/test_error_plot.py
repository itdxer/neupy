import os

import theano

import matplotlib.pyplot as plt
from neupy import plots, algorithms
from neupy.algorithms.base import ErrorHistoryList

from base import BaseTestCase
from data import simple_classification
from utils import (image_comparison, reproducible_network_train,
                   format_image_name, skip_image_comparison_if_specified,
                   catch_stdout)


IMGDIR = os.path.join("plots", "images", "error-plot")


class ErrorPlotTestCase(BaseTestCase):
    def setUp(self):
        super(ErrorPlotTestCase, self).setUp()
        # It's better to use one float type all the time,
        # because different type can change plots images a
        # little bit and this change can cause test failers
        theano.config.floatX = 'float64'

    @skip_image_comparison_if_specified
    def test_error_plot_and_validation_error_warnings(self):
        with catch_stdout() as out:
            network = algorithms.GradientDescent((2, 3, 1), verbose=True)

            network.errors = ErrorHistoryList([1, 2])
            network.validation_errors = ErrorHistoryList([None])

            plots.error_plot(network, ax=None, show=False)
            terminal_output = out.getvalue()
            self.assertIn("error will be ignored", terminal_output)

    @skip_image_comparison_if_specified
    def test_error_plot_ax_none(self):
        ax = plt.gca()

        network = algorithms.GradientDescent((2, 3, 1))
        ax_returned = plots.error_plot(network, ax=None, show=False)

        self.assertIs(ax_returned, ax)

    @skip_image_comparison_if_specified
    def test_error_plot_show_image(self):
        def mock_plt_show():
            pass

        # Test suppose not to fail
        real_plt_show = plt.show
        plt.show = mock_plt_show

        network = reproducible_network_train(step=0.3)
        plots.error_plot(network, show=True)

        plt.show = real_plt_show

    @skip_image_comparison_if_specified
    def test_simple_error_plot(self):
        original_image_name = format_image_name("simple_plot.png")
        original_image = os.path.join(IMGDIR, original_image_name)

        with image_comparison(original_image) as fig:
            ax = fig.add_subplot(1, 1, 1)
            network = reproducible_network_train(step=0.3)
            plots.error_plot(network, ax=ax, show=False)

    @skip_image_comparison_if_specified
    def test_error_plot_with_validation_dataset(self):
        original_image_name = format_image_name("with_validation.png")
        original_image = os.path.join(IMGDIR, original_image_name)

        with image_comparison(original_image) as fig:
            ax = fig.add_subplot(1, 1, 1)

            x_train, x_test, y_train, y_test = simple_classification()
            gdnet = algorithms.GradientDescent((10, 12, 1), step=0.25)
            gdnet.train(x_train, y_train, x_test, y_test, epochs=100)
            plots.error_plot(gdnet, ax=ax, show=False)

    @skip_image_comparison_if_specified
    def test_error_plot_with_log_scale(self):
        original_image_name = format_image_name("log_scale.png")
        original_image = os.path.join(IMGDIR, original_image_name)

        with image_comparison(original_image) as fig:
            ax = fig.add_subplot(1, 1, 1)
            network = reproducible_network_train(step=0.3)
            plots.error_plot(network, logx=True, ax=ax, show=False)
