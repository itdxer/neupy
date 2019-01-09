from neupy import layers
from neupy.exceptions import LayerConnectionError

from base import BaseTestCase


class InputTestCase(BaseTestCase):
    def test_input_exceptions(self):
        error_message = "Input layer got unexpected input"

        with self.assertRaisesRegexp(LayerConnectionError, error_message):
            layers.join(layers.Input(2), layers.Input(1))

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
