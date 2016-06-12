import numpy as np

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
    def test_kohonen_success(self):
        kh = algorithms.Kohonen(
            n_inputs=2,
            n_outputs=3,
            weight=np.array([
                [0.7071, 0.7071, -1.0000],
                [-0.7071, 0.7071,  0.0000],
            ]),
            step=0.5,
            verbose=False,
        )

        # test one iteration update
        data = np.reshape(input_data[0, :], (1, input_data.shape[1]))
        kh.train(data, epochs=1)
        np.testing.assert_array_almost_equal(
            kh.weight,
            np.array([
                [0.7071, 0.4516, -1.0000],
                [-0.7071, 0.84385,  0.0000],
            ]),
            decimal=4
        )

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.Kohonen(
                n_inputs=1,
                n_outputs=2,
                step=0.5,
                verbose=False
            ),
            np.array([1, 2, 3])
        )

    def test_predict_different_inputs(self):
        knet = algorithms.Kohonen(
            n_inputs=1,
            n_outputs=2,
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
