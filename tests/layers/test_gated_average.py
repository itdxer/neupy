import numpy as np

from neupy import layers
from neupy.utils import asfloat, tf_utils
from neupy.exceptions import LayerConnectionError

from base import BaseTestCase


class GatedAverageTestCase(BaseTestCase):
    def test_gated_average_layer_output_shape(self):
        network = layers.join(
            layers.parallel(
                layers.Input(10) >> layers.Softmax(2),
                layers.Input(20) >> layers.Relu(8),
                layers.Input(20) >> layers.Relu(8),
            ),
            layers.GatedAverage()
        )
        self.assertShapesEqual(network.output_shape, (None, 8))

    def test_gated_average_layer_negative_index(self):
        network = layers.join(
            layers.parallel(
                layers.Input(20) >> layers.Relu(8),
                layers.Input(20) >> layers.Relu(8),
                layers.Input(10) >> layers.Softmax(2),
            ),
            layers.GatedAverage(gate_index=-1, name='gate')
        )
        self.assertShapesEqual(network.output_shape, (None, 8))

        network = layers.join(
            layers.parallel(
                layers.Input(10) >> layers.Softmax(2),
                layers.Input(20) >> layers.Relu(8),
                layers.Input(20) >> layers.Relu(8),
            ),
            layers.GatedAverage(gate_index=-3, name='gate')
        )
        self.assertShapesEqual(network.output_shape, (None, 8))

    def test_gated_average_layer_exceptions_index_position(self):
        networks = layers.parallel(
            layers.Input(10) >> layers.Softmax(2),
            layers.Input(20) >> layers.Relu(8),
            layers.Input(20) >> layers.Relu(8),
        )
        with self.assertRaisesRegexp(LayerConnectionError, "Invalid index"):
            layers.join(networks, layers.GatedAverage(gate_index=3))

        with self.assertRaisesRegexp(LayerConnectionError, "Invalid index"):
            layers.join(networks, layers.GatedAverage(gate_index=-4))

    def test_gated_average_layer_exceptions(self):
        networks = layers.parallel(
            layers.Input((10, 3, 3)),
            layers.Input(20) >> layers.Relu(8),
            layers.Input(20) >> layers.Relu(8),
        )
        error_message = "should be 2-dimensional"
        with self.assertRaisesRegexp(LayerConnectionError, error_message):
            layers.join(networks, layers.GatedAverage())

        networks = layers.parallel(
            layers.Input(10) >> layers.Softmax(3),
            layers.Input(20) >> layers.Relu(8),
            layers.Input(20) >> layers.Relu(8),
        )
        error_message = "only 3 networks, got 2 networks"
        with self.assertRaisesRegexp(LayerConnectionError, error_message):
            layers.join(networks, layers.GatedAverage())

        networks = layers.parallel(
            layers.Input(10) >> layers.Softmax(2),
            layers.Input(20) >> layers.Relu(8),
            layers.Input(20) >> layers.Relu(10),
        )
        error_message = "expect to have the same shapes"
        with self.assertRaisesRegexp(LayerConnectionError, error_message):
            layers.join(networks, layers.GatedAverage())

    def test_gated_average_layer_non_default_index(self):
        network = layers.join(
            layers.parallel(
                layers.Input(20) >> layers.Relu(8),
                layers.Input(10) >> layers.Softmax(2),
                layers.Input(20) >> layers.Relu(8),
            ),
            layers.GatedAverage(gate_index=1),
        )
        self.assertShapesEqual(network.output_shape, (None, 8))

    def test_gated_average_layer_output(self):
        network = layers.join(
            layers.Input(10),
            layers.parallel(
                layers.Softmax(2),
                layers.Relu(8),
                layers.Relu(8),
            ),
            layers.GatedAverage(),
        )

        random_input = asfloat(np.random.random((20, 10)))
        actual_output = self.eval(network.output(random_input))
        self.assertShapesEqual(actual_output.shape, (20, 8))

    def test_gated_average_layer_multi_dimensional_inputs(self):
        network = layers.join(
            layers.Input((5, 5, 1)),
            layers.parallel(
                layers.Reshape() >> layers.Softmax(2),
                layers.Convolution((2, 2, 3)),
                layers.Convolution((2, 2, 3)),
            ),
            layers.GatedAverage(),
        )

        self.assertShapesEqual(network.input_shape, (None, 5, 5, 1))
        self.assertShapesEqual(network.output_shape, (None, 4, 4, 3))

        random_input = asfloat(np.random.random((8, 5, 5, 1)))
        actual_output = self.eval(network.output(random_input))

        self.assertEqual(actual_output.shape, (8, 4, 4, 3))
