import os

import pytest
import numpy as np
import matplotlib.pyplot as plt

from neupy import plots

from base import BaseTestCase


IMGDIR = os.path.join("plots", "images", "hinton")


class HintonDiagramTestCase(BaseTestCase):
    single_thread = True

    @pytest.mark.mpl_image_compare
    def test_simple_hinton(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.sca(ax)  # To test the case when ax=None

        weight = np.random.randn(20, 20)
        plots.hinton(weight, add_legend=True)

        return fig

    @pytest.mark.mpl_image_compare
    def test_max_weight(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        weight = 100 * np.random.randn(20, 20)
        plots.hinton(weight, ax=ax, max_weight=10, add_legend=True)

        return fig

    @pytest.mark.mpl_image_compare
    def test_hinton_without_legend(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        weight = np.random.randn(20, 20)
        plots.hinton(weight, ax=ax, add_legend=False)

        return fig

    @pytest.mark.mpl_image_compare
    def test_hinton_only_positive(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        weight = np.random.random((20, 20))
        plots.hinton(weight, ax=ax)

        return fig

    @pytest.mark.mpl_image_compare
    def test_hinton_only_negative(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        weight = -np.random.random((20, 20))
        plots.hinton(weight, ax=ax)

        return fig

    @pytest.mark.mpl_image_compare
    def test_hinton_1darray(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        weight = -np.random.randn(20)
        plots.hinton(weight, ax=ax)

        return fig
