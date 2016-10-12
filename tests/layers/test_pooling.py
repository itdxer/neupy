from collections import namedtuple

import theano
import theano.tensor as T
import numpy as np

from neupy import layers
from neupy.utils import asfloat
from neupy.layers.connections import LayerConnectionError

from base import BaseTestCase


class PoolingLayersTestCase(BaseTestCase):
    use_sandbox_mode = False

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

    def test_upscale_layer_exceptions(self):
        with self.assertRaises(LayerConnectionError):
            # Input shape should have 3 feature dimensions
            # (and +1 for the batch)
            upscale_layer = layers.Upscale((2, 2))
            layers.Input(10) > upscale_layer
            upscale_layer.output_shape

        invalid_scales = [-1, (2, 0), (-4, 1), (3, 3, 3)]
        for invalid_scale in invalid_scales:
            with self.assertRaises(ValueError):
                layers.Upscale(invalid_scale)

    def test_upscale_layer_shape(self):
        Case = namedtuple("Case", "scale expected_shape")
        testcases = (
            Case(scale=(2, 2), expected_shape=(1, 28, 28)),
            Case(scale=(2, 1), expected_shape=(1, 28, 14)),
            Case(scale=(1, 2), expected_shape=(1, 14, 28)),
            Case(scale=(1, 1), expected_shape=(1, 14, 14)),
            Case(scale=(1, 10), expected_shape=(1, 14, 140)),
        )

        for testcase in testcases:
            upscale_layer = layers.Upscale(testcase.scale)
            layers.Input((1, 14, 14)) > upscale_layer

            self.assertEqual(upscale_layer.output_shape,
                             testcase.expected_shape,
                             msg="scale: {}".format(testcase.scale))

    def test_upscale_layer(self):
        input_value = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]).reshape((1, 1, 2, 4))
        expected_output = np.array([
            [1, 1, 2, 2, 3, 3, 4, 4],
            [1, 1, 2, 2, 3, 3, 4, 4],
            [1, 1, 2, 2, 3, 3, 4, 4],
            [5, 5, 6, 6, 7, 7, 8, 8],
            [5, 5, 6, 6, 7, 7, 8, 8],
            [5, 5, 6, 6, 7, 7, 8, 8],
        ]).reshape((1, 1, 6, 8))

        upscale_layer = layers.Upscale((3, 2))
        layers.Input((1, 2, 4)) > upscale_layer

        x = T.tensor4('x')
        actual_output = upscale_layer.output(x)
        actual_output = actual_output.eval({x: asfloat(input_value)})

        np.testing.assert_array_almost_equal(
            asfloat(expected_output),
            actual_output
        )

    def test_global_pooling_output_shape(self):
        input_layer = layers.Input((3, 8, 8))
        global_pooling_layer = layers.GlobalPooling()

        layers.join(input_layer, global_pooling_layer)
        self.assertEqual(global_pooling_layer.output_shape, (3,))

    def test_global_pooling(self):
        x = asfloat(np.ones((2, 3, 4, 5)))
        expected_outputs = np.ones((2, 3))

        global_mena_pooling_layer = layers.GlobalPooling()
        actual_output = global_mena_pooling_layer.output(x).eval()

        self.assertEqual(actual_output.shape, (2, 3))
        np.testing.assert_array_equal(expected_outputs, actual_output)

    def test_global_pooling_other_function(self):
        x = asfloat(np.ones((2, 3, 4, 5)))
        expected_outputs = 20 * np.ones((2, 3))

        global_sum_pooling_layer = layers.GlobalPooling(function=T.sum)
        a = T.tensor4()
        actual_output = global_sum_pooling_layer.output(a).eval({a:x})

        self.assertEqual(actual_output.shape, (2, 3))
        np.testing.assert_array_equal(expected_outputs, actual_output)
