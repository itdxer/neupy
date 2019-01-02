import random
from itertools import product
from collections import namedtuple

import numpy as np
import tensorflow as tf

from neupy import layers
from neupy.utils import asfloat, as_tuple
from neupy.layers.convolutions import conv_output_shape, deconv_output_shape
from neupy.exceptions import LayerConnectionError

from base import BaseTestCase


class ConvLayersTestCase(BaseTestCase):
    def get_shape(self, value):
        shape = self.eval(tf.shape(value))
        return tuple(shape)

    def test_convolution_params(self):
        inp = layers.Input((5, 5, 1))
        conv = layers.Convolution((2, 2, 6))

        # Propagate data through the network in
        # order to trigger initialization
        (inp >> conv).outputs

        self.assertEqual((2, 2, 1, 6), self.get_shape(conv.weight))
        self.assertEqual((6,), self.get_shape(conv.bias))

    def test_conv_shapes(self):
        paddings = ['valid', 'same']
        strides = [(1, 1), (2, 1), (2, 2)]
        x = asfloat(np.random.random((20, 12, 11, 2)))

        for stride, padding in product(strides, paddings):
            inp = layers.Input((12, 11, 2))
            conv = layers.Convolution(
                (3, 4, 5), padding=padding, stride=stride)

            network = inp >> conv
            y = self.eval(network.output(x))
            actual_output_shape = as_tuple(y.shape[1:])

            self.assertEqual(
                actual_output_shape,
                network.output_shape,
                msg='padding={} and stride={}'.format(padding, stride),
            )

    def test_valid_strides(self):
        Case = namedtuple("Case", "stride expected_output")
        testcases = (
            Case(stride=(4, 4), expected_output=(4, 4)),
            Case(stride=(4,), expected_output=(4, 1)),
            Case(stride=4, expected_output=(4, 4)),
        )

        for testcase in testcases:
            conv = layers.Convolution(
                (2, 3, 1), stride=testcase.stride)

            msg = "Input stride size: {}".format(testcase.stride)
            self.assertEqual(
                testcase.expected_output, conv.stride, msg=msg)

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
                layers.Convolution((2, 3, 1), stride=stride)

    def test_valid_padding(self):
        valid_paddings = ('VALID', 'SAME', 'same', 'valid', 10, 1, (7, 1))
        for padding in valid_paddings:
            layers.Convolution((2, 3, 1), padding=padding)

    def test_invalid_padding(self):
        invalid_paddings = ('invalid mode', -10, (10, -5))

        for padding in invalid_paddings:
            msg = "Padding: {}".format(padding)

            with self.assertRaises(ValueError, msg=msg):
                layers.Convolution((2, 3, 1), padding=padding)

    def test_conv_output_shape_func_exceptions(self):
        with self.assertRaises(ValueError):
            # Wrong stride value
            conv_output_shape(
                dimension_size=5, filter_size=5,
                padding='VALID', stride='not int')

        with self.assertRaises(ValueError):
            # Wrong filter size value
            conv_output_shape(
                dimension_size=5, filter_size='not int',
                padding='SAME', stride=5)

        with self.assertRaisesRegexp(ValueError, "unknown \S+ padding value"):
            # Wrong padding value
            conv_output_shape(
                dimension_size=5, filter_size=5,
                padding=1.5, stride=5,
            )
#
    def test_conv_output_shape_int_padding(self):
        output_shape = conv_output_shape(
            dimension_size=10,
            padding=3,
            filter_size=5,
            stride=5,
        )
        self.assertEqual(output_shape, 3)
#
    def test_conv_unknown_dim_size(self):
        shape = conv_output_shape(
            dimension_size=None, filter_size=5,
            padding='VALID', stride=5,
        )
        self.assertEqual(shape, None)
#
    def test_conv_invalid_padding_exception(self):
        error_msg = "greater or equal to zero"
        with self.assertRaisesRegexp(ValueError, error_msg):
            layers.Convolution((1, 3, 3), padding=-1)

        error_msg = "Tuple .+ greater or equal to zero"
        with self.assertRaisesRegexp(ValueError, error_msg):
            layers.Convolution((1, 3, 3), padding=(2, -1))

        with self.assertRaisesRegexp(ValueError, "invalid string value"):
            layers.Convolution((1, 3, 3), padding='NOT_SAME')

        with self.assertRaisesRegexp(ValueError, "contains two elements"):
            layers.Convolution((1, 3, 3), padding=(3, 3, 3))

    def test_conv_invalid_input_shape(self):
        inp = layers.Input(10)
        conv = layers.Convolution((1, 3, 3))
        network = inp >> conv

        with self.assertRaises(LayerConnectionError):
            network.outputs

    def test_conv_with_custom_int_padding(self):
        inp = layers.Input((5, 5, 1))
        conv = layers.Convolution((3, 3, 1), bias=0, weight=1, padding=2)

        network = (inp >> conv)
        network.outputs

        x = asfloat(np.ones((1, 5, 5, 1)))
        expected_output = np.array([
            [1, 2, 3, 3, 3, 2, 1],
            [2, 4, 6, 6, 6, 4, 2],
            [3, 6, 9, 9, 9, 6, 3],
            [3, 6, 9, 9, 9, 6, 3],
            [3, 6, 9, 9, 9, 6, 3],
            [2, 4, 6, 6, 6, 4, 2],
            [1, 2, 3, 3, 3, 2, 1],
        ]).reshape((1, 7, 7, 1))

        actual_output = self.eval(network.output(x))
        np.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_conv_with_custom_tuple_padding(self):
        inp = layers.Input((5, 5, 1))
        conv = layers.Convolution((3, 3, 1), bias=0, weight=1, padding=(0, 2))

        network = (inp >> conv)
        network.outputs

        x = asfloat(np.ones((1, 5, 5, 1)))
        expected_output = np.array([
            [3, 6, 9, 9, 9, 6, 3],
            [3, 6, 9, 9, 9, 6, 3],
            [3, 6, 9, 9, 9, 6, 3],
        ]).reshape((1, 3, 7, 1))
        actual_output = self.eval(network.output(x))

        np.testing.assert_array_almost_equal(expected_output, actual_output)
        self.assertEqual(network.output_shape, (3, 7, 1))

    def test_conv_without_bias(self):
        inp = layers.Input((5, 5, 1))
        conv = layers.Convolution((3, 3, 1), bias=None, weight=1)

        network = inp >> conv
        network.outputs

        x = asfloat(np.ones((1, 5, 5, 1)))
        expected_output = 9 * np.ones((1, 3, 3, 1))
        actual_output = self.eval(network.output(x))

        np.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_conv_unknown_input_width_and_height(self):
        network = layers.join(
            layers.Input((None, None, 3)),
            layers.Convolution((3, 3, 5)),
        )
        self.assertEqual(network.output_shape, (None, None, 5))

        input_value = asfloat(np.ones((1, 12, 12, 3)))
        actual_output = self.eval(network.output(input_value))
        self.assertEqual(actual_output.shape, (1, 10, 10, 5))

        input_value = asfloat(np.ones((1, 21, 21, 3)))
        actual_output = self.eval(network.output(input_value))
        self.assertEqual(actual_output.shape, (1, 19, 19, 5))

    def test_dilated_convolution(self):
        network = layers.join(
            layers.Input((6, 6, 1)),
            layers.Convolution((3, 3, 1), dilation=2, weight=1, bias=None),
        )

        input_value = asfloat(np.arange(36).reshape(1, 6, 6, 1))
        actual_output = self.eval(network.output(input_value))

        self.assertEqual(actual_output.shape, (1, 2, 2, 1))
        self.assertEqual(actual_output.shape[1:], network.output_shape)

        actual_output = actual_output[0, :, :, 0]
        expected_output = np.array([
            [126, 135],  # every row value adds +1 per filter value (+9)
            [180, 189],  # every col value adds +6 per filter value (+54)
        ])
        np.testing.assert_array_almost_equal(actual_output, expected_output)


class DeconvolutionTestCase(BaseTestCase):
    def test_deconvolution(self):
        network = layers.join(
            layers.Input((10, 10, 3)),
            layers.Convolution((3, 3, 7)),
            layers.Deconvolution((3, 3, 4)),
        )

        shapes = network.output_shapes_per_layer
        self.assertDictEqual(
            shapes, {
                network.layers[0]: (10, 10, 3),
                network.layers[1]: (8, 8, 7),
                network.layers[2]: (10, 10, 4),
            }
        )

        input_value = asfloat(np.random.random((1, 10, 10, 3)))
        actual_output = self.eval(network.output(input_value))

        self.assertEqual(actual_output.shape, (1, 10, 10, 4))

    def test_deconvolution_same_padding(self):
        network = layers.join(
            layers.Input((10, 10, 3)),
            layers.Convolution((3, 3, 7), padding='same'),
            layers.Deconvolution((3, 3, 4), padding='same'),
        )

        shapes = network.output_shapes_per_layer
        self.assertDictEqual(
            shapes, {
                network.layers[0]: (10, 10, 3),
                network.layers[1]: (10, 10, 7),
                network.layers[2]: (10, 10, 4),
            }
        )

        input_value = asfloat(np.random.random((1, 10, 10, 3)))
        actual_output = self.eval(network.output(input_value))

        self.assertEqual(actual_output.shape, (1, 10, 10, 4))

    def test_deconvolution_int_padding(self):
        network = layers.join(
            layers.Input((10, 10, 3)),
            layers.Convolution((3, 3, 7), padding=9),
            layers.Deconvolution((3, 3, 4), padding=9),
        )

        shapes = network.output_shapes_per_layer
        self.assertDictEqual(
            shapes, {
                network.layers[0]: (10, 10, 3),
                network.layers[1]: (26, 26, 7),
                network.layers[2]: (10, 10, 4),
            }
        )

        input_value = asfloat(np.random.random((1, 10, 10, 3)))
        actual_output = self.eval(network.output(input_value))

        self.assertEqual(actual_output.shape, (1, 10, 10, 4))

    def test_deconvolution_tuple_padding(self):
        network = layers.join(
            layers.Input((10, 10, 3)),
            layers.Convolution((3, 3, 7), padding=(9, 3)),
            layers.Deconvolution((3, 3, 4), padding=(9, 3)),
        )

        shapes = network.output_shapes_per_layer
        self.assertSequenceEqual(
            shapes, {
                network.layers[0]: (10, 10, 3),
                network.layers[1]: (26, 14, 7),
                network.layers[2]: (10, 10, 4),
            }
        )

        input_value = asfloat(np.random.random((1, 10, 10, 3)))
        actual_output = self.eval(network.output(input_value))

        self.assertEqual(actual_output.shape, (1, 10, 10, 4))

    def test_deconv_unknown_input_width_and_height(self):
        network = layers.join(
            layers.Input((None, None, 3)),
            layers.Convolution((3, 3, 7)),
            layers.Deconvolution((3, 3, 4)),
        )

        shapes = network.output_shapes_per_layer
        self.assertDictEqual(
            shapes, {
                network.layers[0]: (None, None, 3),
                network.layers[1]: (None, None, 7),
                network.layers[2]: (None, None, 4),
            }
        )

        input_value = asfloat(np.random.random((1, 10, 10, 3)))
        actual_output = self.eval(network.output(input_value))
        self.assertEqual(actual_output.shape, (1, 10, 10, 4))

        input_value = asfloat(np.random.random((1, 7, 7, 3)))
        actual_output = self.eval(network.output(input_value))
        self.assertEqual(actual_output.shape, (1, 7, 7, 4))

    def test_deconv_output_shape(self):
        self.assertEqual(None, deconv_output_shape(None, 3, 'same', 1))

        self.assertEqual(12, deconv_output_shape(10, 3, 'valid', 1))
        self.assertEqual(16, deconv_output_shape(10, 7, 'valid', 1))
        self.assertEqual(10, deconv_output_shape(10, 3, 'same', 1))

        self.assertEqual(14, deconv_output_shape(4, 5, 'valid', 3))
        self.assertEqual(12, deconv_output_shape(4, 3, 'same', 3))
        self.assertEqual(12, deconv_output_shape(4, 7, 'same', 3))

    def test_deconv_output_shape_exception(self):
        with self.assertRaisesRegexp(ValueError, "unknown \S+ padding"):
            deconv_output_shape(10, 3, padding='xxx', stride=1)

        with self.assertRaisesRegexp(ValueError, "doesn't support dilation"):
            deconv_output_shape(10, 3, padding='valid', stride=1, dilation=2)

    def test_deconvolution_for_random_cases(self):
        # A few random cases will check if output shape computed from
        # the network is the same as the shape that we get after we
        # propagated input through the network.
        for test_id in range(30):
            width = random.randint(7, 20)
            height = random.randint(7, 20)

            fh = random.randint(1, 7)
            fw = random.randint(1, 7)

            pad = random.choice([
                'valid',
                'same',
                random.randint(0, 10),
                (
                    random.randint(0, 10),
                    random.randint(0, 10),
                ),
            ])
            stride = random.choice([
                random.randint(1, 4),
                (
                    random.randint(1, 4),
                    random.randint(1, 4),
                ),
            ])

            print('\n------------')
            print("Test case #{}".format(test_id))
            print('------------')
            print("Image shape: {}x{}".format(height, width))
            print("Filter shape: {}x{}".format(fh, fw))
            print("Padding: {}".format(pad))
            print("Stride: {}".format(stride))

            network = layers.join(
                layers.Input((height, width, 1)),
                layers.Convolution((fh, fw, 2), padding=pad, stride=stride),
                layers.Deconvolution((fh, fw, 1), padding=pad, stride=stride),
            )

            input_value = asfloat(np.random.random((1, height, width, 1)))
            actual_output = self.eval(network.output(input_value))
            self.assertEqual(actual_output.shape[1:], network.output_shape)
