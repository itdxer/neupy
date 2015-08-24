import numpy as np

from neuralpy.algorithms import Oja

from base import BaseTestCase


class OjaTestCase(BaseTestCase):
    def setUp(self):
        super(OjaTestCase, self).setUp()
        self.data = np.array([
            [2, 2],
            [1, 1],
            [4, 4],
            [5, 5],
        ])

    def test_oja_minimization(self):
        nw = Oja(
            minimized_data_size=1,
            step=0.01,
            weights=np.ones((2, 1)) * 0.1
        )
        result = np.array([
            [2.83],
            [1.41],
            [5.66],
            [7.07],
        ])

        minimized_data = nw.train(self.data, epsilon=1e-5)
        self.assertTrue(np.all(np.round(minimized_data, 2) == result))

        reconstructed = nw.reconstruct(minimized_data)
        self.assertTrue(np.allclose(reconstructed, self.data))

    def test_oja_exceptions(self):
        nw = Oja(minimized_data_size=1, step=0.01)

        with self.assertRaises(ValueError):
            # Can't reconstruct without training
            nw.reconstruct(np.random.random((4, 1)))

        nw.train(self.data, epsilon=1e-5)

        with self.assertRaises(ValueError):
            # Invalid #feature for reconstruct
            nw.reconstruct(np.random.random((3, 3)))

        with self.assertRaises(ValueError):
            # Invalid #feature for train
            nw.train(np.random.random((4, 10)))
