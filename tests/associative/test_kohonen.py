import numpy as np

from neupy.layers import LinearLayer, CompetitiveOutputLayer
from neupy import algorithms

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
        kh = algorithms.Kohonen(self.conn, step=0.5, verbose=False)

        # test one iteration update
        data = np.reshape(input_data[0, :], (1, input_data.shape[1]))
        kh.train(data, epochs=1)
        self.assertTrue(np.all(
            kh.input_layer.weight == np.array([
                [0.7071, 0.4516, -1.0000],
                [-0.7071, 0.84385,  0.0000],
            ])
        ))

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.Kohonen(
                LinearLayer(1) > CompetitiveOutputLayer(2),
                step=0.5,
                verbose=False
            ),
            np.array([1, 2, 3])
        )

    def test_predict_different_inputs(self):
        knet = algorithms.Kohonen(
            LinearLayer(1) > CompetitiveOutputLayer(2),
            step=0.5,
            verbose=False,
        )

        data = np.array([[1, 1, 1]]).T
        target = np.array([
            [1, 0],
            [1, 0],
            [1, 0],
        ])

        knet.train(data, epochs=100)
        self.assertInvalidVectorPred(knet, data.ravel(), target,
                                     decimal=2)
