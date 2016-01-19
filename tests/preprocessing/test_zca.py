import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

from neupy import preprocessing
from neupy.utils import NotTrainedException

from base import BaseTestCase
from utils import image_comparison


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PLOTS_DIR = os.path.join(CURRENT_DIR, "plots", "zca")
IMGDIR = os.path.join(CURRENT_DIR, "images")


class ZCATestCase(BaseTestCase):
    def test_exceptions(self):
        with self.assertRaises(NotTrainedException):
            preprocessing.ZCA().transform(np.random.random((3, 2)))

    def test_simple_zca(self):
        plt.style.use('ggplot')

        original_image = os.path.join(PLOTS_DIR, "test_simple_zca.png")
        image = os.path.join(IMGDIR, "cifar10.png")

        data = imread(image)
        data = data[:, :, 0]

        with image_comparison(original_image, figsize=(10, 6)) as fig:
            ax = fig.add_subplot(1, 1, 1)

            zca = preprocessing.ZCA(0.001)
            zca.train(data)
            data_transformed = zca.transform(data)

            ax.imshow(data_transformed, cmap=plt.cm.binary)
