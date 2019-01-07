import numpy as np

from neupy import layers
from neupy.utils import asfloat
from neupy.exceptions import LayerConnectionError

from base import BaseTestCase


class GatedAverageTestCase(BaseTestCase):
    def test_gated_average_layer_negative_index(self):
        gated_avg_layer = layers.GatedAverage(gating_layer_index=-1)
        layers.join([
            layers.Input(20) >> layers.Relu(8),
            layers.Input(20) >> layers.Relu(8),
            layers.Input(10) >> layers.Softmax(2),
        ], gated_avg_layer)

        self.assertEqual(gated_avg_layer.output_shape, (8,))
        self.assertEqual(gated_avg_layer.input_shape, [(8,), (8,), (2,)])

        gated_avg_layer = layers.GatedAverage(gating_layer_index=-3)
        layers.join([
            layers.Input(10) >> layers.Softmax(2),
            layers.Input(20) >> layers.Relu(8),
            layers.Input(20) >> layers.Relu(8),
        ], gated_avg_layer)

        self.assertEqual(gated_avg_layer.output_shape, (8,))
        self.assertEqual(gated_avg_layer.input_shape, [(2,), (8,), (8,)])

    def test_gated_average_layer_exceptions_index_position(self):
        gated_avg_layer = layers.GatedAverage(gating_layer_index=3)
        with self.assertRaisesRegexp(LayerConnectionError, "Invalid index"):
            layers.join([
                layers.Input(20) >> layers.Relu(8),
                layers.Input(10) >> layers.Softmax(2),
                layers.Input(20) >> layers.Relu(8),
            ], gated_avg_layer)

        gated_avg_layer = layers.GatedAverage(gating_layer_index=-4)
        with self.assertRaisesRegexp(LayerConnectionError, "Invalid index"):
            layers.join([
                layers.Input(10) >> layers.Softmax(2),
                layers.Input(20) >> layers.Relu(8),
                layers.Input(20) >> layers.Relu(8),
            ], gated_avg_layer)

    def test_gated_average_layer_exceptions(self):
        gated_avg_layer = layers.GatedAverage()
        with self.assertRaisesRegexp(LayerConnectionError, "should be vector"):
            layers.join([
                layers.Input((10, 3, 3)),  # shape not 1d
                layers.Input(20) >> layers.Relu(8),
                layers.Input(20) >> layers.Relu(8),
            ], gated_avg_layer)

        gated_avg_layer = layers.GatedAverage()
        error_message = "only 3 networks, got 2 networks"
        with self.assertRaisesRegexp(LayerConnectionError, error_message):
            layers.join([
                layers.Input(10) >> layers.Softmax(3),
                layers.Input(20) >> layers.Relu(8),
                layers.Input(20) >> layers.Relu(8),
            ], gated_avg_layer)

        gated_avg_layer = layers.GatedAverage()
        error_message = "expect to have the same shapes"
        with self.assertRaisesRegexp(LayerConnectionError, error_message):
            layers.join([
                layers.Input(10) >> layers.Softmax(2),
                layers.Input(20) >> layers.Relu(8),
                layers.Input(20) >> layers.Relu(10),
            ], gated_avg_layer)

    def test_gated_average_layer_non_default_index(self):
        gated_avg_layer = layers.GatedAverage(gating_layer_index=1)
        layers.join([
            layers.Input(20) >> layers.Relu(8),
            layers.Input(10) >> layers.Softmax(2),
            layers.Input(20) >> layers.Relu(8),
        ], gated_avg_layer)

        self.assertEqual(gated_avg_layer.output_shape, (8,))
        self.assertEqual(gated_avg_layer.input_shape, [(8,), (2,), (8,)])

    def test_gated_average_layer_output_shape(self):
        gated_avg_layer = layers.GatedAverage()
        self.assertIsNone(gated_avg_layer.output_shape)

        layers.join([
            layers.Input(10) >> layers.Softmax(2),
            layers.Input(20) >> layers.Relu(8),
            layers.Input(20) >> layers.Relu(8),
        ], gated_avg_layer)

        self.assertEqual(gated_avg_layer.output_shape, (8,))
        self.assertEqual(gated_avg_layer.input_shape, [(2,), (8,), (8,)])

    def test_gated_average_layer_output(self):
        input_layer = layers.Input(10)
        network = layers.join(
            [
                input_layer >> layers.Softmax(2),
                input_layer >> layers.Relu(8),
                input_layer >> layers.Relu(8),
            ],
            layers.GatedAverage()
        )

        random_input = asfloat(np.random.random((20, 10)))
        actual_output = self.eval(network.output(random_input))

        self.assertEqual(actual_output.shape, (20, 8))

    def test_gated_average_layer_multi_dimensional_inputs(self):
        input_layer = layers.Input((5, 5, 1))
        network = layers.join(
            [
                input_layer >> layers.Reshape() >> layers.Softmax(2),
                input_layer >> layers.Convolution((2, 2, 3)),
                input_layer >> layers.Convolution((2, 2, 3)),
            ],
            layers.GatedAverage()
        )

        self.assertEqual(network.input_shape, (5, 5, 1))
        self.assertEqual(network.output_shape, (4, 4, 3))

        random_input = asfloat(np.random.random((8, 5, 5, 1)))
        actual_output = self.eval(network.output(random_input))

        self.assertEqual(actual_output.shape, (8, 4, 4, 3))
