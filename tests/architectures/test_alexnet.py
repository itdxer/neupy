import numpy as np
import tensorflow as tf

from neupy import architectures
from neupy.utils import asfloat
from neupy.architectures.alexnet import SliceChannels

from base import BaseTestCase


class AlexnetTestCase(BaseTestCase):
    def test_alexnet_architecture(self):
        alexnet = architectures.alexnet()
        self.assertEqual(alexnet.input_shape, (3, 227, 227))
        self.assertEqual(alexnet.output_shape, (1000,))

        random_input = asfloat(np.random.random((7, 3, 227, 227)))
        prediction = self.eval(alexnet.output(random_input))
        self.assertEqual(prediction.shape, (7, 1000))

    def test_alexnet_slicing_layer_repr(self):
        sc_layer = SliceChannels(0, 30)
        self.assertEqual(repr(sc_layer), "SliceChannels(0, 30)")

    def test_alexnet_slicing_layer(self):
        sc_layer = SliceChannels(0, 30)
        test_input = tf.Variable(
            asfloat(np.arange(60).reshape((1, -1, 1, 1))),
            dtype=tf.float32,
        )

        actual_result = self.eval(sc_layer.output(test_input))
        expected_result = asfloat(np.arange(30).reshape((1, -1, 1, 1)))

        np.testing.assert_array_almost_equal(actual_result, expected_result)
