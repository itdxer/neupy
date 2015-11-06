import numpy as np

from neupy import plots

from base import BaseTestCase
from utils import image_comparison


class HintonDiagramTestCase(BaseTestCase):
    def test_simple_hinton(self):
        original_image = "plots/images/test_simple_hinton.png"
        with image_comparison(original_image, figsize=(16, 12)) as fig:
            weight = np.random.randn(20, 20)
            ax = fig.add_subplot(1, 1, 1)
            plots.hinton(weight, ax=ax)
