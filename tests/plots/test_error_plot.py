import os
import warnings

import matplotlib.pyplot as plt
from neupy import plots, algorithms, layers

from base import BaseTestCase
from data import simple_classification
from utils import (
    image_comparison, reproducible_network_train,
    format_image_name, skip_image_comparison_if_specified,
)


IMGDIR = os.path.join("plots", "images", "error-plot")


class ErrorPlotTestCase(BaseTestCase):
    single_thread = True

    @skip_image_comparison_if_specified
    def test_error_plot_and_validation_error_warnings(self):
        with warnings.catch_warnings(record=True) as warns:
            network = algorithms.GradientDescent(
                layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1),
                verbose=True,
                batch_size=None,
            )

            network.training_errors = [1, 2]
            network.validation_errors = [None]

            plots.error_plot(network, ax=None, show=False)

            self.assertEqual(len(warns), 1)
            self.assertIn("error will be ignored", str(warns[0].message))

    @skip_image_comparison_if_specified
    def test_error_plot_ax_none(self):
        ax = plt.gca()

        network = algorithms.GradientDescent(
            layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1),
            batch_size=None,
        )
        ax_returned = plots.error_plot(network, ax=None, show=False)

        self.assertIs(ax_returned, ax)

    @skip_image_comparison_if_specified
    def test_error_plot_show_image(self):
        # Just making sure that nothing will fail
        real_plt_show = plt.show
        plt.show = lambda: None

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
            gdnet = algorithms.GradientDescent(
                [
                    layers.Input(10),
                    layers.Sigmoid(12),
                    layers.Sigmoid(1),
                ],
                step=0.25,
                batch_size=None,
            )
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
