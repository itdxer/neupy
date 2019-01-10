from collections import namedtuple

import numpy as np
import tensorflow as tf

from neupy import layers
from neupy.utils import asfloat
from neupy.exceptions import LayerConnectionError
from neupy.layers.pooling import pooling_output_shape

from base import BaseTestCase


class PoolingLayersTestCase(BaseTestCase):
    def test_pooling_output_shape_exception(self):
        expected_msg = r"1 is unknown padding value for pooling"
        with self.assertRaisesRegexp(ValueError, expected_msg):
            pooling_output_shape(
                dimension_size=5,
                pool_size=2,
                padding=1,
                stride=2)

    def test_pooling_output_shape(self):
        otuput_shape = pooling_output_shape(None, None, None, None)
        self.assertEqual(otuput_shape, None)

        otuput_shape = pooling_output_shape(
            dimension_size=5, pool_size=2,
            padding='VALID', stride=2)

        self.assertEqual(otuput_shape, 2)

        otuput_shape = pooling_output_shape(
            dimension_size=5, pool_size=2,
            padding='VALID', stride=1)

        self.assertEqual(otuput_shape, 4)

        otuput_shape = pooling_output_shape(
            dimension_size=5, pool_size=2,
            padding='VALID', stride=4)

        self.assertEqual(otuput_shape, 1)

    def test_pooling_undefined_output_shape(self):
        max_pool_layer = layers.MaxPooling((2, 2))
        self.assertShapesEqual(
            max_pool_layer.output_shape,
            (None, None, None, None))

    def test_pooling_layer_output_shape(self):
        network = layers.join(
            layers.Input((10, 10, 3)),
            layers.MaxPooling((2, 2)),
        )
        self.assertShapesEqual(network.output_shape, (None, 5, 5, 3))

    def test_pooling_stride_int(self):
        network = layers.join(
            layers.Input((28, 28, 1)),
            layers.MaxPooling((2, 2), stride=1, padding='same'),
        )
        self.assertShapesEqual(
            network.input_shape,
            network.output_shape)

    def test_pooling_exceptions(self):
        with self.assertRaises(ValueError):
            layers.MaxPooling((2, 2), padding='TEST')

        with self.assertRaises(ValueError):
            layers.MaxPooling((2, 2), padding=1)

    def test_pooling_repr(self):
        layer = layers.MaxPooling((2, 2))
        self.assertEqual(
            str(layer),
            (
                "MaxPooling((2, 2), stride=None, padding='valid', "
                "name='max-pooling-1')"
            ),
        )
        layer = layers.MaxPooling((2, 2), stride=(3, 3))
        self.assertEqual(
            str(layer),
            (
                "MaxPooling((2, 2), stride=(3, 3), padding='valid', "
                "name='max-pooling-2')"
            ),
        )

    def test_max_pooling(self):
        X = asfloat(np.array([
            [1, 2, 3, -1],
            [4, -6, 3, 1],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
        ])).reshape(1, 4, 4, 1)
        expected_output = asfloat(np.array([
            [4, 3],
            [0, 1],
        ])).reshape(1, 2, 2, 1)

        network = layers.join(
            layers.Input((4, 4, 1)),
            layers.MaxPooling((2, 2)),
        )
        actual_output = self.eval(network.output(X))
        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_average_pooling(self):
        X = asfloat(np.array([
            [1, 2, 3, -1],
            [4, -6, 3, 1],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
        ])).reshape(1, 4, 4, 1)
        expected_output = asfloat(np.array([
            [1 / 4., 6 / 4.],
            [-1 / 4., 1 / 4.],
        ])).reshape(1, 2, 2, 1)

        network = layers.join(
            layers.Input((4, 4, 1)),
            layers.AveragePooling((2, 2)),
        )
        actual_output = self.eval(network.output(X))
        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_upscale_late_shape_init(self):
        network = layers.join(
            layers.AveragePooling((2, 2)),
            layers.Softmax(),
        )
        self.assertShapesEqual(network.output_shape, (None, None, None, None))

        network = layers.join(layers.Input((10, 10, 1)), network)
        self.assertShapesEqual(network.output_shape, (None, 5, 5, 1))


class UpscaleLayersTestCase(BaseTestCase):
    def test_upscale_layer_exceptions(self):
        with self.assertRaises(LayerConnectionError):
            # Input shape should be 4 dimensional
            layers.join(layers.Input(10), layers.Upscale((2, 2)))

        invalid_scales = [-1, (2, 0), (-4, 1), (3, 3, 3)]
        for invalid_scale in invalid_scales:
            with self.assertRaises(ValueError):
                layers.Upscale(invalid_scale)

    def test_upscale_layer_shape(self):
        Case = namedtuple("Case", "scale expected_shape")
        testcases = (
            Case(scale=(2, 2), expected_shape=(None, 28, 28, 1)),
            Case(scale=(2, 1), expected_shape=(None, 28, 14, 1)),
            Case(scale=(1, 2), expected_shape=(None, 14, 28, 1)),
            Case(scale=(1, 1), expected_shape=(None, 14, 14, 1)),
            Case(scale=(1, 10), expected_shape=(None, 14, 140, 1)),
        )

        for testcase in testcases:
            network = layers.join(
                layers.Input((14, 14, 1)),
                layers.Upscale(testcase.scale),
            )

            self.assertShapesEqual(
                network.output_shape,
                testcase.expected_shape,
                msg="scale: {}".format(testcase.scale))

    def test_upscale_layer(self):
        input_value = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]).reshape((1, 2, 4, 1))
        expected_output = np.array([
            [1, 1, 2, 2, 3, 3, 4, 4],
            [1, 1, 2, 2, 3, 3, 4, 4],
            [1, 1, 2, 2, 3, 3, 4, 4],
            [5, 5, 6, 6, 7, 7, 8, 8],
            [5, 5, 6, 6, 7, 7, 8, 8],
            [5, 5, 6, 6, 7, 7, 8, 8],
        ]).reshape((1, 6, 8, 1))

        upscale_layer = layers.Upscale((3, 2))
        network = layers.join(layers.Input((2, 4, 1)), upscale_layer)
        self.assertShapesEqual(network.output_shape, (None, 6, 8, 1))

        actual_output = self.eval(network.output(asfloat(input_value)))
        np.testing.assert_array_almost_equal(
            asfloat(expected_output), actual_output)

    def test_upscale_late_shape_init(self):
        network = layers.join(
            layers.Upscale((2, 2)),
            layers.Softmax(),
        )
        self.assertShapesEqual(network.output_shape, (None, None, None, None))

        network = layers.join(layers.Input((10, 10, 5)), network)
        self.assertShapesEqual(network.output_shape, (None, 20, 20, 5))

    def test_global_pooling_repr(self):
        self.assertEqual(
            "Upscale((2, 3), name='upscale-1')",
            str(layers.Upscale((2, 3))))


class GlobalPoolingLayersTestCase(BaseTestCase):
    def test_global_pooling_output_shape(self):
        input_layer = layers.Input((8, 8, 3))
        global_pooling_layer = layers.GlobalPooling('avg')
        network = layers.join(
            input_layer,
            global_pooling_layer
        )
        self.assertShapesEqual(network.input_shape, (None, 8, 8, 3))
        self.assertShapesEqual(network.output_shape, (None, 3))

    def test_global_pooling(self):
        x = asfloat(np.ones((2, 4, 5, 3)))
        expected_outputs = np.ones((2, 3))

        global_mena_pooling_layer = layers.GlobalPooling('avg')
        actual_output = self.eval(global_mena_pooling_layer.output(x))

        self.assertEqual(actual_output.shape, (2, 3))
        np.testing.assert_array_equal(expected_outputs, actual_output)

    def test_global_pooling_other_function(self):
        x = asfloat(np.ones((2, 4, 5, 3)))
        expected_outputs = 20 * np.ones((2, 3))

        global_sum_pooling_layer = layers.GlobalPooling(function=tf.reduce_sum)
        actual_output = self.eval(global_sum_pooling_layer.output(x))

        self.assertShapesEqual(actual_output.shape, (2, 3))
        np.testing.assert_array_equal(expected_outputs, actual_output)

    def test_global_pooling_unknown_option(self):
        with self.assertRaises(ValueError):
            layers.GlobalPooling('unknown')

    def test_global_pooling_for_lower_dimensions(self):
        layer = layers.GlobalPooling('max')
        x = np.random.random((4, 5))
        np.testing.assert_array_equal(x, layer.output(x))

    def test_global_pooling_late_shape_init(self):
        network = layers.join(
            layers.Convolution((3, 3, 12)),
            layers.GlobalPooling('max'),
        )
        self.assertShapesEqual(network.output_shape, (None, None))

        network = layers.join(layers.Input((10, 10, 1)), network)
        self.assertShapesEqual(network.output_shape, (None, 12))

    def test_global_pooling_repr(self):
        layer = layers.GlobalPooling('max')
        self.assertEqual(
            "GlobalPooling('max', name='global-pooling-1')",
            str(layer))

        layer = layers.GlobalPooling(lambda x: x)
        self.assertRegexpMatches(
            str(layer),
            r"GlobalPooling\(<function .+>, name='global-pooling-2'\)",
        )
