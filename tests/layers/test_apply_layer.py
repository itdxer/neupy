import numpy as np
import tensorflow as tf

from neupy import layers

from base import BaseTestCase


class ApplyLayerTestCase(BaseTestCase):
    def test_basic_apply_layer(self):
        network = layers.Input(5) >> layers.Apply(lambda x: x / 4)  # noqa

        self.assertShapesEqual(network.input_shape, (None, 5))
        self.assertShapesEqual(network.output_shape, (None, 5))

        input_array = np.ones((17, 5))
        actual_output = network.predict(input_array)
        expected_output = input_array / 4

        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_apply_layer_as_activation(self):
        input = np.random.random((100, 5))

        base_network = layers.Input(5) >> layers.Linear(10)
        network_1 = base_network >> layers.Relu()
        network_2 = base_network >> layers.Apply(tf.nn.relu)

        np.testing.assert_array_almost_equal(network_1.predict(input), network_2.predict(input))
