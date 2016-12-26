import numpy as np

from neupy import layers

from base import BaseTestCase


class ReshapeLayerTestCase(BaseTestCase):
    def test_reshape_layer_1d_shape(self):
        x = np.random.random((5, 4, 3, 2, 1))

        input_layer = layers.Input((4, 3, 2, 1))
        reshape_layer = layers.Reshape()
        input_layer > reshape_layer

        y = reshape_layer.output(x).eval()
        self.assertEqual(y.shape, (5, 4 * 3 * 2 * 1))

    def test_reshape_layer_2d_shape(self):
        x = np.random.random((5, 20))

        input_layer = layers.Input(20)
        reshape_layer = layers.Reshape((4, 5))
        input_layer > reshape_layer

        y = reshape_layer.output(x).eval()
        self.assertEqual(y.shape, (5, 4, 5))
