import os

import numpy as np
import theano
import matplotlib.pyplot as plt

from neupy import plots

from base import BaseTestCase
from utils import (skip_image_comparison_if_specified, image_comparison,
                   format_image_name)


IMGDIR = os.path.join("plots", "images", "hinton")


class HintonDiagramTestCase(BaseTestCase):
    def setUp(self):
        super(HintonDiagramTestCase, self).setUp()
        theano.config.floatX = 'float64'

    @skip_image_comparison_if_specified
    def test_simple_hinton(self):
        original_image_name = format_image_name("simple_hinton.png")
        original_image = os.path.join(IMGDIR, original_image_name)

        with image_comparison(original_image, figsize=(10, 6)) as fig:
            weight = np.random.randn(20, 20)
            ax = fig.add_subplot(1, 1, 1)
            plt.sca(ax)  # To test the case when ax=None
            plots.hinton(weight, add_legend=True)

    @skip_image_comparison_if_specified
    def test_max_weight(self):
        original_image_name = format_image_name("max_weight_hinton.png")
        original_image = os.path.join(IMGDIR, original_image_name)

        with image_comparison(original_image, figsize=(10, 6)) as fig:
            weight = 100 * np.random.randn(20, 20)
            ax = fig.add_subplot(1, 1, 1)
            plots.hinton(weight, ax=ax, max_weight=10, add_legend=True)

    @skip_image_comparison_if_specified
    def test_hinton_without_legend(self):
        original_image_name = format_image_name("hinton_without_legend.png")
        original_image = os.path.join(IMGDIR, original_image_name)

        with image_comparison(original_image, figsize=(10, 6)) as fig:
            weight = np.random.randn(20, 20)
            ax = fig.add_subplot(1, 1, 1)
            plots.hinton(weight, ax=ax, add_legend=False)

    @skip_image_comparison_if_specified
    def test_hinton_only_positive(self):
        original_image_name = format_image_name("hinton_only_positive.png")
        original_image = os.path.join(IMGDIR, original_image_name)

        with image_comparison(original_image, figsize=(10, 6)) as fig:
            weight = np.random.random((20, 20))
            ax = fig.add_subplot(1, 1, 1)
            plots.hinton(weight, ax=ax)

    @skip_image_comparison_if_specified
    def test_hinton_only_negative(self):
        original_image_name = format_image_name("hinton_only_negative.png")
        original_image = os.path.join(IMGDIR, original_image_name)

        with image_comparison(original_image, figsize=(10, 6)) as fig:
            weight = -np.random.random((20, 20))
            ax = fig.add_subplot(1, 1, 1)
            plots.hinton(weight, ax=ax)

    @skip_image_comparison_if_specified
    def test_hinton_1darray(self):
        original_image_name = format_image_name("hinton_1darray.png")
        original_image = os.path.join(IMGDIR, original_image_name)

        with image_comparison(original_image, figsize=(10, 4)) as fig:
            weight = -np.random.randn(20)
            ax = fig.add_subplot(1, 1, 1)
            plots.hinton(weight, ax=ax)
