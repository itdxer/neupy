from neupy import layers
from neupy.exceptions import LayerConnectionError

from base import BaseTestCase


class InputTestCase(BaseTestCase):
    def test_input_exceptions(self):
        layer = layers.Input(10)
        error_message = "Input layer got unexpected input"

        with self.assertRaisesRegexp(LayerConnectionError, error_message):
            layer.get_output_shape((10, 15))

    def test_output_shape_merging(self):
        layer = layers.Input(10)
        self.assertShapesEqual(layer.get_output_shape((None, 10)), (None, 10))
        self.assertShapesEqual(layer.get_output_shape((5, 10)), (5, 10))

        layer = layers.Input((None, None, 3))
        self.assertShapesEqual(
            layer.get_output_shape((None, 28, 28, 3)),
            (None, 28, 28, 3),
        )
        self.assertShapesEqual(
            layer.get_output_shape((None, None, 28, 3)),
            (None, None, 28, 3),
        )
        self.assertShapesEqual(
            layer.get_output_shape((10, 28, 28, None)),
            (10, 28, 28, 3),
        )

    def test_merged_inputs(self):
        network = layers.join(
            layers.Input((10, 2)),
            layers.Input((None, 2)),
        )
        self.assertShapesEqual(network.input_shape, (None, 10, 2))
        self.assertShapesEqual(network.output_shape, (None, 10, 2))

    def test_input_layers_connected(self):
        network = layers.join(layers.Input(1), layers.Input(1))
        self.assertShapesEqual(network.input_shape, (None, 1))
        self.assertShapesEqual(network.output_shape, (None, 1))

    def test_input_repr(self):
        self.assertEqual(
            str(layers.Input(10)),
            "Input(10, name='input-1')",
        )
        self.assertEqual(
            str(layers.Input((10, 3))),
            "Input((10, 3), name='input-2')",
        )
        self.assertEqual(
            str(layers.Input((None, None, 3))),
            "Input((None, None, 3), name='input-3')",
        )
        self.assertEqual(
            str(layers.Input(None)),
            "Input(None, name='input-4')",
        )

    def test_input_with_tensor_shape(self):
        network = layers.join(
            layers.Input(10),
            layers.Relu(5),
        )
        network_2 = layers.join(
            layers.Input(network.output_shape[1:]),
            layers.Relu(3),
        )
        self.assertEqual(network_2.layers[0].shape, (5,))
        self.assertShapesEqual(network_2.input_shape, (None, 5))
        self.assertShapesEqual(network_2.output_shape, (None, 3))
