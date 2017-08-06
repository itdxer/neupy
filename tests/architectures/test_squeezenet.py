import numpy as np

from neupy.utils import asfloat
from neupy import architectures

from base import BaseTestCase


class SqueezenetTestCase(BaseTestCase):
    def test_squeezenet_architecture(self):
        squeezenet = architectures.squeezenet()
        self.assertEqual(squeezenet.input_shape, (3, 224, 224))
        self.assertEqual(squeezenet.output_shape, (1000,))

        squeezenet_predict = squeezenet.compile()

        random_input = asfloat(np.random.random((7, 3, 224, 224)))
        prediction = squeezenet_predict(random_input)
        self.assertEqual(prediction.shape, (7, 1000))
