import numpy as np

from neupy.utils import asfloat
from neupy import architectures

from base import BaseTestCase


class VGG16TestCase(BaseTestCase):
    def test_vgg16_architecture(self):
        vgg16 = architectures.vgg16()
        self.assertEqual(vgg16.input_shape, (3, 224, 224))
        self.assertEqual(vgg16.output_shape, (1000,))

        random_input = asfloat(np.random.random((7, 3, 224, 224)))
        prediction = self.eval(vgg16.output(random_input))
        self.assertEqual(prediction.shape, (7, 1000))
