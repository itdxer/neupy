from itertools import product
from collections import namedtuple

import theano
import theano.tensor as T
import numpy as np

from neupy.utils import asfloat, as_tuple
from neupy import layers
from neupy.layers.connections import LayerConnectionError
from neupy.layers.convolutions import conv_output_shape

from base import BaseTestCase


class ConvLayersTestCase(BaseTestCase):
    def test_convolution_params(self):
        weight_shape = (6, 1, 2, 2)
        bias_shape = (6,)

        input_layer = layers.Input((1, 5, 5))
        conv_layer = layers.Convolution((6, 2, 2))

        input_layer > conv_layer
        conv_layer.initialize()

        self.assertEqual(weight_shape, conv_layer.weight.get_value().shape)
        self.assertEqual(bias_shape, conv_layer.bias.get_value().shape)

    def test_conv_shapes(self):
        border_modes = [
            'valid', 'full', 'half',
            4, 5,
            (6, 3), (4, 4), (1, 1)
        ]
        strides = [(1, 1), (2, 1), (2, 2)]
        x = asfloat(np.random.random((20, 2, 12, 11)))

        for stride, border_mode in product(strides, border_modes):
            input_layer = layers.Input((2, 12, 11))
            conv_layer = layers.Convolution((5, 3, 4),
                                            border_mode=border_mode,
                                            stride_size=stride)

            input_layer > conv_layer
            conv_layer.initialize()

            y = conv_layer.output(x).eval()
            actual_output_shape = as_tuple(y.shape[1:])

            self.assertEqual(actual_output_shape, conv_layer.output_shape,
                             msg='border_mode={}'.format(border_mode))

    def test_valid_strides(self):
        Case = namedtuple("Case", "stride_size expected_output")
        testcases = (
            Case(stride_size=(4, 4), expected_output=(4, 4)),
            Case(stride_size=(4,), expected_output=(4, 1)),
            Case(stride_size=4, expected_output=(4, 4)),
        )

        for testcase in testcases:
            conv_layer = layers.Convolution((1, 2, 3),
                                            stride_size=testcase.stride_size)
            msg = "Input stride size: {}".format(testcase.stride_size)
            self.assertEqual(testcase.expected_output, conv_layer.stride_size,
                             msg=msg)

    def test_invalid_strides(self):
        invalid_strides = (
            (4, 4, 4),
            -10,
            (-5, -5),
            (-5, 5),
            (-5, 0),
        )

        for stride_size in invalid_strides:
            msg = "Input stride size: {}".format(stride_size)
            with self.assertRaises(ValueError, msg=msg):
                layers.Convolution((1, 2, 3), stride_size=stride_size)

    def test_valid_border_mode(self):
        valid_border_modes = ('valid', 'full', 'half', (5, 3), 4, (4, 0))
        for border_mode in valid_border_modes:
            layers.Convolution((1, 2, 3), border_mode=border_mode)

    def test_invalid_border_mode(self):
        invalid_border_modes = ('invalid mode', -10, (10, -5))

        for border_mode in invalid_border_modes:
            msg = "Input border mode: {}".format(border_mode)
            with self.assertRaises(ValueError, msg=msg):
                layers.Convolution((1, 2, 3), border_mode=border_mode)

    def test_conv_output_shape_func_exceptions(self):
        with self.assertRaises(ValueError):
            conv_output_shape(dimension_size=5, filter_size=5, border_mode=5,
                              stride='not int')

        with self.assertRaises(ValueError):
            conv_output_shape(dimension_size=5, filter_size='not int',
                              border_mode=5, stride=5)

        with self.assertRaises(ValueError):
            conv_output_shape(dimension_size=5, filter_size=5,
                              border_mode='invalid value', stride=5)


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
