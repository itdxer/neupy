from neupy import layers

from base import BaseTestCase


class CreateCustomLayersTestCase(BaseTestCase):
    def test_custom_layer(self):
        class CustomLayer(layers.BaseLayer):
            def output(self, input):
                return 2 * input

        custom_layer = CustomLayer()
        network = layers.join(layers.Input(10), custom_layer)

        self.assertShapesEqual(network.input_shape, (None, 10))
        self.assertShapesEqual(network.output_shape, None)

        custom_layer.input_shape = (None, 10)
        err_message = "Cannot update input shape of the layer"
        with self.assertRaisesRegexp(ValueError, err_message):
            custom_layer.input_shape = (None, 20)

    def test_custom_layer_repr_priority(self):
        class CustomLayer(layers.Identity):
            def __init__(self, value, name=None):
                self.value = value
                super(CustomLayer, self).__init__(name=name)

            def __repr__(self):
                return self._repr_arguments(self.value, a=3, name=self.name)

        custom_layer = CustomLayer(33)
        self.assertEqual(
            repr(custom_layer),
            "CustomLayer(33, name='custom-layer-1', a=3)")
