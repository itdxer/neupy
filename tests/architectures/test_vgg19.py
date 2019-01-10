import numpy as np

from neupy.utils import asfloat
from neupy import architectures

from base import BaseTestCase


class VGG19TestCase(BaseTestCase):
    def test_vgg19_architecture(self):
        vgg19 = architectures.vgg19()
        self.assertShapesEqual(vgg19.input_shape, (None, 224, 224, 3))
        self.assertShapesEqual(vgg19.output_shape, (None, 1000))

        random_input = asfloat(np.random.random((7, 224, 224, 3)))
        prediction = self.eval(vgg19.output(random_input))
        self.assertEqual(prediction.shape, (7, 1000))
