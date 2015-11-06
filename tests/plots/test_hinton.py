import numpy as np

from neupy import plots

from base import BaseTestCase
from utils import image_comparison


class HintonDiagramTestCase(BaseTestCase):
    def test_simple_hinton(self):
        original_image = "plots/images/test_simple_hinton.png"
        with image_comparison(original_image, figsize=(10, 6)) as fig:
            weight = np.random.randn(20, 20)
            ax = fig.add_subplot(1, 1, 1)
            plots.hinton(weight, ax=ax, add_legend=True)

    def test_max_weight(self):
        original_image = "plots/images/test_max_weight_hinton.png"
        with image_comparison(original_image, figsize=(10, 6)) as fig:
            weight = 100 * np.random.randn(20, 20)
            ax = fig.add_subplot(1, 1, 1)
            plots.hinton(weight, ax=ax, max_weight=10, add_legend=True)

    def test_hinton_without_legend(self):
        original_image = "plots/images/test_hinton_without_legend.png"
        with image_comparison(original_image, figsize=(10, 6)) as fig:
            weight = np.random.randn(20, 20)
            ax = fig.add_subplot(1, 1, 1)
            plots.hinton(weight, ax=ax, add_legend=False)

    def test_hinton_only_positive(self):
        original_image = "plots/images/test_hinton_only_positive.png"
        with image_comparison(original_image, figsize=(10, 6)) as fig:
            weight = np.random.random((20, 20))
            ax = fig.add_subplot(1, 1, 1)
            plots.hinton(weight, ax=ax)

    def test_hinton_only_negative(self):
        original_image = "plots/images/test_hinton_only_negative.png"
        with image_comparison(original_image, figsize=(10, 6)) as fig:
            weight = -np.random.random((20, 20))
            ax = fig.add_subplot(1, 1, 1)
            plots.hinton(weight, ax=ax)

    def test_hinton_1darray(self):
        original_image = "plots/images/test_hinton_1darray.png"
        with image_comparison(original_image, figsize=(10, 4)) as fig:
            weight = -np.random.randn(20)
            ax = fig.add_subplot(1, 1, 1)
            plots.hinton(weight, ax=ax)
