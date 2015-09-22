import numpy as np

from neupy.layers import LinearLayer, CompetitiveOutputLayer
from neupy.algorithms import Kohonen

from base import BaseTestCase


input_data = np.array([
    [0.1961,  0.9806],
    [-0.1961,  0.9806],
    [0.9806,  0.1961],
    [0.9806, -0.1961],
    [-0.5812, -0.8137],
    [-0.8137, -0.5812],
])


class KohonenTestCase(BaseTestCase):
    def setUp(self):
        super(KohonenTestCase, self).setUp()
        weight = np.array([
            [0.7071, 0.7071, -1.0000],
            [-0.7071, 0.7071,  0.0000],
        ])
        input_layer = LinearLayer(2, weight=weight)
        output_layer = CompetitiveOutputLayer(3)
        self.conn = input_layer > output_layer

    def test_kohonen_success(self):
        kh = Kohonen(self.conn, step=0.5)

        # test one iteration update
        data = np.reshape(input_data[0, :], (1, input_data.shape[1]))
        kh.train(data, epochs=1)
        self.assertTrue(np.all(
            kh.input_layer.weight == np.array([
                [0.7071, 0.4516, -1.0000],
                [-0.7071, 0.84385,  0.0000],
            ])
        ))
