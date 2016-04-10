import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

from neupy import preprocessing
from neupy.utils import NotTrainedException

from base import BaseTestCase
from utils import (image_comparison, format_image_name,
                   skip_image_comparison_if_specified)


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, "plots", "zca")
IMGDIR = os.path.join(CURRENT_DIR, "images")


class ZCATestCase(BaseTestCase):
    def test_exceptions(self):
        with self.assertRaises(NotTrainedException):
            data = np.random.random((3, 2))
            zca = preprocessing.ZCA()
            zca.transform(data)

    @skip_image_comparison_if_specified
    def test_simple_zca(self):
        plt.style.use('ggplot')

        original_image_name = format_image_name("simple_zca.png")
        original_image = os.path.join(PLOTS_DIR, original_image_name)
        image = os.path.join(IMGDIR, "cifar10.png")

        data = imread(image)
        data = data[:, :, 0]

        comparison_kwargs = dict(figsize=(10, 6), tol=0.05)

        with image_comparison(original_image, **comparison_kwargs) as fig:
            ax = fig.add_subplot(1, 1, 1)

            zca = preprocessing.ZCA(0.001)
            zca.train(data)
            data_transformed = zca.transform(data)

            ax.imshow(data_transformed, cmap=plt.cm.binary)

        with image_comparison(original_image, **comparison_kwargs) as fig:
            ax = fig.add_subplot(1, 1, 1)

            zca = preprocessing.ZCA(0.001)
            data_transformed = zca.fit(data).transform(data)

            ax.imshow(data_transformed, cmap=plt.cm.binary)
