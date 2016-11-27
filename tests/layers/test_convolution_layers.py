from itertools import product
from collections import namedtuple

import numpy as np

from neupy import layers
from neupy.utils import asfloat, as_tuple
from neupy.layers.convolutions import conv_output_shape
from neupy.exceptions import LayerConnectionError

from base import BaseTestCase


class ConvLayersTestCase(BaseTestCase):
    def test_convolution_params(self):
        weight_shape = (6, 1, 2, 2)
        bias_shape = (6,)

        input_layer = layers.Input((1, 5, 5))
        conv_layer = layers.Convolution((6, 2, 2))

        self.assertEqual(conv_layer.output_shape, None)

        input_layer > conv_layer
        conv_layer.initialize()

        self.assertEqual(weight_shape, conv_layer.weight.get_value().shape)
        self.assertEqual(bias_shape, conv_layer.bias.get_value().shape)

    def test_conv_shapes(self):
        paddings = [
            'valid', 'full', 'half',
            4, 5,
            (6, 3), (4, 4), (1, 1)
        ]
        strides = [(1, 1), (2, 1), (2, 2)]
        x = asfloat(np.random.random((20, 2, 12, 11)))

        for stride, padding in product(strides, paddings):
            input_layer = layers.Input((2, 12, 11))
            conv_layer = layers.Convolution((5, 3, 4),
                                            padding=padding,
                                            stride=stride)

            input_layer > conv_layer
            conv_layer.initialize()

            y = conv_layer.output(x).eval()
            actual_output_shape = as_tuple(y.shape[1:])

            self.assertEqual(actual_output_shape, conv_layer.output_shape,
                             msg='padding={}'.format(padding))

    def test_valid_strides(self):
        Case = namedtuple("Case", "stride expected_output")
        testcases = (
            Case(stride=(4, 4), expected_output=(4, 4)),
            Case(stride=(4,), expected_output=(4, 1)),
            Case(stride=4, expected_output=(4, 4)),
        )

        for testcase in testcases:
            conv_layer = layers.Convolution((1, 2, 3),
                                            stride=testcase.stride)
            msg = "Input stride size: {}".format(testcase.stride)
            self.assertEqual(testcase.expected_output, conv_layer.stride,
                             msg=msg)

    def test_conv_invalid_strides(self):
        invalid_strides = (
            (4, 4, 4),
            -10,
            (-5, -5),
            (-5, 5),
            (-5, 0),
        )

        for stride in invalid_strides:
            msg = "Input stride size: {}".format(stride)
            with self.assertRaises(ValueError, msg=msg):
                layers.Convolution((1, 2, 3), stride=stride)

    def test_valid_padding(self):
        valid_paddings = ('valid', 'full', 'half', (5, 3), 4, (4, 0))
        for padding in valid_paddings:
            layers.Convolution((1, 2, 3), padding=padding)

    def test_invalid_padding(self):
        invalid_paddings = ('invalid mode', -10, (10, -5))

        for padding in invalid_paddings:
            msg = "Input border mode: {}".format(padding)
            with self.assertRaises(ValueError, msg=msg):
                layers.Convolution((1, 2, 3), padding=padding)

    def test_conv_output_shape_func_exceptions(self):
        with self.assertRaises(ValueError):
            conv_output_shape(dimension_size=5, filter_size=5, padding=5,
                              stride='not int')

        with self.assertRaises(ValueError):
            conv_output_shape(dimension_size=5, filter_size='not int',
                              padding=5, stride=5)

        with self.assertRaises(ValueError):
            conv_output_shape(dimension_size=5, filter_size=5,
                              padding='invalid value', stride=5)

    def test_conv_unknown_dim_size(self):
        shape = conv_output_shape(dimension_size=None, filter_size=5,
                                  padding=5, stride=5)
        self.assertEqual(shape, None)

    def test_conv_invalid_padding_exception(self):
        with self.assertRaises(ValueError):
            layers.Convolution((1, 3, 3), padding=(3, 3, 3))

    def test_conv_invalid_input_shape(self):
        conv = layers.Convolution((1, 3, 3))
        with self.assertRaises(LayerConnectionError):
            layers.join(layers.Input(10), conv)

    def test_conv_without_bias(self):
        input_layer = layers.Input((1, 5, 5))
        conv = layers.Convolution((1, 3, 3), bias=None, weight=1)

        connection = input_layer > conv
        connection.initialize()

        x = asfloat(np.ones((1, 1, 5, 5)))
        expected_output = 9 * np.ones((1, 1, 3, 3))
        actual_output = connection.output(x).eval()

        np.testing.assert_array_almost_equal(expected_output, actual_output)
