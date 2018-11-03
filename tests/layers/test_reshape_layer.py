import numpy as np

from neupy import layers
from neupy.exceptions import LayerConnectionError

from base import BaseTestCase


class ReshapeLayerTestCase(BaseTestCase):
    def test_reshape_layer_1d_shape(self):
        x = np.random.random((5, 4, 3, 2, 1))

        input_layer = layers.Input((4, 3, 2, 1))
        reshape_layer = layers.Reshape()
        input_layer > reshape_layer

        y = self.eval(reshape_layer.output(x))
        self.assertEqual(y.shape, (5, 4 * 3 * 2 * 1))

    def test_reshape_layer_2d_shape(self):
        x = np.random.random((5, 20))

        input_layer = layers.Input(20)
        reshape_layer = layers.Reshape((4, 5))
        input_layer > reshape_layer

        y = self.eval(reshape_layer.output(x))
        self.assertEqual(y.shape, (5, 4, 5))

    def test_reshape_unknown_shape(self):
        conn = layers.join(
            layers.Input((None, 20)),
            layers.Reshape(),
        )

        x = np.random.random((7, 12, 20))
        y = self.eval(conn.output(x))
        self.assertEqual(y.shape, (7, 12 * 20))

    def test_reshape_with_negative_value(self):
        conn = layers.join(
            layers.Input((7, 20)),
            layers.Reshape([5, -1]),
        )
        self.assertEqual(conn.output_shape, (5, 28))

        x = np.random.random((11, 7, 20))
        y = self.eval(conn.output(x))
        self.assertEqual(y.shape, (11, 5, 28))

    def test_reshape_with_negative_value_unknown_in_shape(self):
        conn = layers.join(
            layers.Input((7, None)),
            layers.Reshape([5, -1]),
        )
        self.assertEqual(conn.output_shape, (5, None))

        x = np.random.random((11, 7, 10))
        y = self.eval(conn.output(x))
        self.assertEqual(y.shape, (11, 5, 14))

    def test_reshape_exceptions(self):
        with self.assertRaisesRegexp(ValueError, "Only single"):
            layers.Reshape([-1, -1])

        with self.assertRaisesRegexp(ValueError, "Cannot derive"):
            layers.join(
                layers.Input(20),
                layers.Reshape((-1, 6)),
                layers.Reshape(),  # this layer uses output_shape
            )


class TransposeTestCase(BaseTestCase):
    def test_simple_transpose(self):
        conn = layers.join(
            layers.Input((7, 11)),
            layers.Transpose([2, 1]),
        )
        self.assertEqual(conn.output_shape, (11, 7))

    def test_transpose_unknown_input_dim(self):
        conn = layers.join(
            layers.Input((None, 10, 20)),
            layers.Transpose([2, 1, 3]),
        )
        self.assertEqual(conn.output_shape, (10, None, 20))

    def test_transpose_exceptions(self):
        with self.assertRaisesRegexp(ValueError, "cannot be used"):
            layers.join(
                layers.Input((7, 11)),
                layers.Transpose([2, 0]),  # cannot use 0 index (batch dim)
            )

        with self.assertRaisesRegexp(LayerConnectionError, "at least 3"):
            layers.join(
                layers.Input(20),
                layers.Transpose([2, 1]),
            )
