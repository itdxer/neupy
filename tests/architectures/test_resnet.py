import numpy as np

from neupy.utils import asfloat
from neupy import architectures

from base import BaseTestCase


class Resnet50TestCase(BaseTestCase):
    def test_resnet50_architecture(self):
        resnet50 = architectures.resnet50()
        self.assertEqual(resnet50.input_shape, (3, 224, 224))
        self.assertEqual(resnet50.output_shape, (1000,))

        random_input = asfloat(np.random.random((7, 3, 224, 224)))
        prediction = self.eval(resnet50.output(random_input))
        self.assertEqual(prediction.shape, (7, 1000))
