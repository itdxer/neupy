import numpy as np

from neupy import plots, layers
from neupy.exceptions import InvalidConnection

from base import BaseTestCase


class SaliencyMapTestCase(BaseTestCase):
    single_thread = True

    def setUp(self):
        super(SaliencyMapTestCase, self).setUp()
        self.network = layers.join(
            layers.Input((28, 28, 3)),
            layers.Convolution((3, 3, 8), name='conv') >> layers.Relu(),
            layers.Reshape(),
            layers.Softmax(10),
        )
        self.image = np.ones((28, 28, 3))

    def test_saliency_map_invalid_mode(self):
        message = "'invalid-mode' is invalid value for mode argument"

        with self.assertRaisesRegexp(ValueError, message):
            plots.saliency_map(self.network, self.image, mode='invalid-mode')

    def test_saliency_map_invalid_n_outputs(self):
        new_network = layers.join(
            self.network,
            layers.parallel(
                layers.Sigmoid(1),
                layers.Sigmoid(2),
            )
        )
        message = (
            "Cannot build saliency map for the network that "
            "has more than one output layer."
        )
        with self.assertRaisesRegexp(InvalidConnection, message):
            plots.saliency_map(new_network, self.image)

    def test_saliency_map_invalid_n_inputs(self):
        new_network = layers.join(
            layers.parallel(
                layers.Input((28, 28, 3)),
                layers.Input((28, 28, 3)),
            ),
            layers.Concatenate(),
            self.network.start('conv'),
        )
        message = (
            "Cannot build saliency map for the network that "
            "has more than one input layer."
        )
        with self.assertRaisesRegexp(InvalidConnection, message):
            plots.saliency_map(new_network, self.image)

    def test_saliency_map_invalid_input_image(self):
        network = layers.join(
            layers.Input(10),
            layers.Relu(),
        )
        message = (
            "Input layer has to be 4 dimensions, but network expects "
            "2 dimensional input"
        )
        with self.assertRaisesRegexp(InvalidConnection, message):
            plots.saliency_map(network, self.image)

        message = (
            "Invalid image shape. Image expected to be 3D, got 2D image"
        )
        with self.assertRaisesRegexp(ValueError, message):
            plots.saliency_map(self.network, np.ones((28, 28)))
