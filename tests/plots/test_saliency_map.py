import numpy as np

from neupy import plots, layers
from neupy.exceptions import InvalidConnection

from base import BaseTestCase


class SaliencyMapTestCase(BaseTestCase):
    def test_invalid_arguments_exceptions(self):
        network = layers.join(
            layers.Input((3, 28, 28)),
            layers.Convolution((8, 3, 3)) > layers.Relu(),
            layers.Reshape(),
            layers.Softmax(10),
        )
        image = np.ones((1, 3, 28, 28))

        with self.assertRaisesRegexp(ValueError, 'Invalid image shape'):
            plots.saliency_map(network, np.ones((28, 28)))

        with self.assertRaisesRegexp(ValueError, 'invalid value'):
            plots.saliency_map(network, image, mode='invalid-mode')

        with self.assertRaises(InvalidConnection):
            new_network = network > [
                layers.Sigmoid(1), layers.Sigmoid(2)
            ]
            plots.saliency_map(new_network, image)
