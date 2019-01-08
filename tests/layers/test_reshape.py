import numpy as np

from neupy import layers
from neupy.utils import asfloat
from neupy.exceptions import LayerConnectionError

from base import BaseTestCase


class ReshapeLayerTestCase(BaseTestCase):
    def test_reshape_layer_1d_shape(self):
        x = np.random.random((5, 4, 3, 2, 1))
        network = layers.Input((4, 3, 2, 1)) >> layers.Reshape()

        y = self.eval(network.output(x))
        self.assertEqual(y.shape, (5, 4 * 3 * 2 * 1))

    def test_reshape_layer_2d_shape(self):
        x = np.random.random((5, 20))

        input_layer = layers.Input(20)
        reshape_layer = layers.Reshape((4, 5))
        input_layer > reshape_layer

        y = self.eval(reshape_layer.output(x))
        self.assertEqual(y.shape, (5, 4, 5))
        self.assertShapesEqual(reshape_layer.output_shape, (None, 4, 5))

    def test_reshape_unknown_shape(self):
        network = layers.join(
            layers.Input((None, 20)),
            layers.Reshape(),
        )
        self.assertShapesEqual(network.output_shape, (None, None))

        x = np.random.random((7, 12, 20))
        y = self.eval(network.output(x))
        self.assertEqual(y.shape, (7, 12 * 20))

    def test_reshape_with_negative_value(self):
        network = layers.join(
            layers.Input((7, 20)),
            layers.Reshape((5, -1)),
        )
        self.assertShapesEqual(network.output_shape, (None, 5, 28))

        x = np.random.random((11, 7, 20))
        y = self.eval(network.output(x))
        self.assertEqual(y.shape, (11, 5, 28))

    def test_reshape_with_negative_value_unknown_in_shape(self):
        network = layers.join(
            layers.Input((7, None)),
            layers.Reshape([5, -1]),
        )
        self.assertShapesEqual(network.output_shape, (None, 5, None))

        x = np.random.random((11, 7, 10))
        y = self.eval(network.output(x))
        self.assertEqual(y.shape, (11, 5, 14))

    def test_reshape_exceptions(self):
        with self.assertRaisesRegexp(ValueError, "Only single"):
            layers.Reshape([-1, -1])

        with self.assertRaisesRegexp(ValueError, "are incompatible"):
            layers.join(
                layers.Input(20),
                layers.Reshape((-1, 6)),
            )


class TransposeTestCase(BaseTestCase):
    def test_simple_transpose(self):
        network = layers.join(
            layers.Input((7, 11)),
            layers.Transpose((0, 2, 1)),
        )
        self.assertShapesEqual(network.output_shape, (None, 11, 7))

    def test_transpose_unknown_input_dim(self):
        network = layers.join(
            layers.Input((None, 10, 20)),
            layers.Transpose((0, 2, 1, 3)),
        )
        self.assertShapesEqual(network.output_shape, (None, 10, None, 20))

        value = asfloat(np.random.random((12, 100, 10, 20)))
        output_value = self.eval(network.output(value))
        self.assertEqual(output_value.shape, (12, 10, 100, 20))

        value = asfloat(np.random.random((12, 33, 10, 20)))
        output_value = self.eval(network.output(value))
        self.assertEqual(output_value.shape, (12, 10, 33, 20))

    def test_transpose_exceptions(self):
        error_message = "Cannot apply transpose operation to the input"
        with self.assertRaisesRegexp(LayerConnectionError, error_message):
            layers.join(
                layers.Input(20),
                layers.Transpose((0, 2, 1)),
            )
