import numpy as np

from neupy.utils import asfloat
from neupy import architectures

from base import BaseTestCase


class VGG19TestCase(BaseTestCase):
    def test_vgg19_architecture(self):
        vgg19 = architectures.vgg19()
        self.assertEqual(vgg19.input_shape, (3, 224, 224))
        self.assertEqual(vgg19.output_shape, (1000,))

        vgg19_predict = vgg19.compile()

        random_input = asfloat(np.random.random((7, 3, 224, 224)))
        prediction = vgg19_predict(random_input)
        self.assertEqual(prediction.shape, (7, 1000))
