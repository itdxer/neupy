import numpy as np

from neupy import layers
from neupy.utils import asfloat

from base import BaseTestCase


class EmbeddingLayerTestCase(BaseTestCase):
    def test_embedding_layer(self):
        weight = np.arange(10).reshape((5, 2))

        input_layer = layers.Input(1)
        embedding_layer = layers.Embedding(5, 2, weight=weight)

        connection = layers.join(input_layer, embedding_layer)
        connection.initialize()

        input_vector = asfloat(np.array([[0, 1, 4]]).T)
        expected_output = np.array([
            [[0, 1]],
            [[2, 3]],
            [[8, 9]],
        ])
        actual_output = connection.output(input_vector).eval()

        self.assertEqual(embedding_layer.output_shape, (1, 2))
        np.testing.assert_array_equal(expected_output, actual_output)

    def test_embedding_layer_repr(self):
        layer = layers.Embedding(5, 2)
        self.assertEqual("Embedding(5, 2)", str(layer))

    def test_embedding_output_shape(self):
        layer = layers.Embedding(5, 2)
        self.assertEqual(layer.output_shape, None)
