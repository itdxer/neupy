import os

import numpy as np

from neupy import plots, algorithms

from base import BaseTestCase
from data import simple_classification
from utils import image_comparison, reproducible_network_train


IMGDIR = os.path.join("plots", "images", "error-plot")


class ErrorPlotTestCase(BaseTestCase):
    def test_simple_plot(self):
        original_image = os.path.join(IMGDIR, "test_simple_plot.png")
        with image_comparison(original_image) as fig:
            ax = fig.add_subplot(1, 1, 1)
            network = reproducible_network_train()
            network.plot_errors(ax=ax, show=False)

    def test_plot_with_validation_dataset(self):
        original_image = os.path.join(IMGDIR, "test_with_validation.png")
        with image_comparison(original_image) as fig:
            ax = fig.add_subplot(1, 1, 1)

            x_train, x_test, y_train, y_test = simple_classification()
            gdnet = algorithms.GradientDescent((10, 12, 1), step=0.25)
            gdnet.train(x_train, y_train, x_test, y_test, epochs=100)
            gdnet.plot_errors(ax=ax, show=False)

    def test_log_scale(self):
        original_image = os.path.join(IMGDIR, "test_log_scale.png")
        with image_comparison(original_image) as fig:
            ax = fig.add_subplot(1, 1, 1)
            network = reproducible_network_train()
            network.plot_errors(logx=True, ax=ax, show=False)
