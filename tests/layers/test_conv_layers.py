import theano
import numpy as np

from neupy.utils import asfloat
from neupy import layers

from base import BaseTestCase


class ConvLayersTestCase(BaseTestCase):
    def test_convolution(self):
        weight_shape = (6, 1, 2, 2)
        conv_layer = layers.Convolution(weight_shape)
        conv_layer.initialize()
        self.assertEqual(weight_shape, conv_layer.weight.get_value().shape)

    def test_max_pooling(self):
        input_data = theano.shared(
            asfloat(np.array([
                [1, 2, 3, -1],
                [4, -6, 3, 1],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
            ]))
        )
        expected_output = asfloat(np.array([
            [4, 3],
            [0, 1],
        ]))

        max_pool_layer = layers.MaxPooling((2, 2))
        actual_output = max_pool_layer.output(input_data).eval()
        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_average_pooling(self):
        input_data = theano.shared(
            asfloat(np.array([
            [1, 2, 3, -1],
            [4, -6, 3, 1],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            ]))
        )
        expected_output = asfloat(np.array([
            [1 / 4., 6 / 4.],
            [-1 / 4., 1 / 4.],
        ]))

        average_pool_layer = layers.AveragePooling((2, 2))
        actual_output = average_pool_layer.output(input_data).eval()
        np.testing.assert_array_almost_equal(actual_output, expected_output)
