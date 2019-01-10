import numpy as np

from neupy.utils import asfloat
from neupy import architectures

from base import BaseTestCase


class VGG16TestCase(BaseTestCase):
    def test_vgg16_architecture(self):
        vgg16 = architectures.vgg16()
        self.assertShapesEqual(vgg16.input_shape, (None, 224, 224, 3))
        self.assertShapesEqual(vgg16.output_shape, (None, 1000))

        random_input = asfloat(np.random.random((2, 224, 224, 3)))
        prediction = self.eval(vgg16.output(random_input))
        self.assertEqual(prediction.shape, (2, 1000))
