import numpy as np

from neupy.utils import asfloat
from neupy import architectures

from base import BaseTestCase


class SqueezenetTestCase(BaseTestCase):
    def test_squeezenet_architecture(self):
        squeezenet = architectures.squeezenet()
        self.assertShapesEqual(squeezenet.input_shape, (None, 227, 227, 3))
        self.assertShapesEqual(squeezenet.output_shape, (None, 1000))

        random_input = asfloat(np.random.random((7, 227, 227, 3)))
        prediction = self.eval(squeezenet.output(random_input))
        self.assertEqual(prediction.shape, (7, 1000))
