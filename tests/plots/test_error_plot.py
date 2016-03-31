import os

import theano
import numpy as np

from neupy import plots, algorithms

from base import BaseTestCase
from data import simple_classification
from utils import (image_comparison, reproducible_network_train,
                   format_image_name, skip_image_comparison_if_specified)


IMGDIR = os.path.join("plots", "images", "error-plot")


class ErrorPlotTestCase(BaseTestCase):
    def setUp(self):
        super(ErrorPlotTestCase, self).setUp()
        theano.config.floatX = 'float64'

    @skip_image_comparison_if_specified
    def test_simple_plot(self):
        original_image_name = format_image_name("simple_plot.png")
        original_image = os.path.join(IMGDIR, original_image_name)

        with image_comparison(original_image) as fig:
            ax = fig.add_subplot(1, 1, 1)
            network = reproducible_network_train(step=0.3)
            network.plot_errors(ax=ax, show=False)

    @skip_image_comparison_if_specified
    def test_plot_with_validation_dataset(self):
        original_image_name = format_image_name("with_validation.png")
        original_image = os.path.join(IMGDIR, original_image_name)

        with image_comparison(original_image) as fig:
            ax = fig.add_subplot(1, 1, 1)

            x_train, x_test, y_train, y_test = simple_classification()
            gdnet = algorithms.GradientDescent((10, 12, 1), step=0.25)
            gdnet.train(x_train, y_train, x_test, y_test, epochs=100)
            gdnet.plot_errors(ax=ax, show=False)

    @skip_image_comparison_if_specified
    def test_log_scale(self):
        original_image_name = format_image_name("log_scale.png")
        original_image = os.path.join(IMGDIR, original_image_name)

        with image_comparison(original_image) as fig:
            ax = fig.add_subplot(1, 1, 1)
            network = reproducible_network_train(step=0.3)
            network.plot_errors(logx=True, ax=ax, show=False)
