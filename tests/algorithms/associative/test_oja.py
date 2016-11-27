import numpy as np

from neupy import algorithms, init
from neupy.exceptions import NotTrained

from base import BaseTestCase
from utils import vectors_for_testing


class OjaTestCase(BaseTestCase):
    def setUp(self):
        super(OjaTestCase, self).setUp()
        self.data = np.array([
            [2, 2],
            [1, 1],
            [4, 4],
            [5, 5],
        ])
        self.result = np.array([
            [2.83],
            [1.41],
            [5.66],
            [7.07],
        ])

    def test_oja_minimization(self):
        ojanet = algorithms.Oja(
            minimized_data_size=1,
            step=0.01,
            weight=init.Constant(0.1),
            verbose=False
        )

        ojanet.train(self.data, epsilon=1e-5, epochs=100)
        minimized_data = ojanet.predict(self.data)
        np.testing.assert_array_almost_equal(
            minimized_data, self.result,
            decimal=2
        )

        reconstructed = ojanet.reconstruct(minimized_data)
        np.testing.assert_array_almost_equal(
            reconstructed, self.data,
            decimal=3
        )

    def test_oja_exceptions(self):
        ojanet = algorithms.Oja(minimized_data_size=1, step=0.01,
                                verbose=False)

        with self.assertRaises(NotTrained):
            # Can't reconstruct without training
            ojanet.reconstruct(np.random.random((4, 1)))

        with self.assertRaises(NotTrained):
            # Can't predict without training
            ojanet.predict(np.random.random((4, 1)))

        ojanet.train(self.data, epsilon=1e-5)

        with self.assertRaises(ValueError):
            # Invalid #feature for reconstruct
            ojanet.reconstruct(np.random.random((3, 3)))

        with self.assertRaises(ValueError):
            # Invalid #feature for train
            ojanet.train(np.random.random((4, 10)))

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.Oja(minimized_data_size=1, verbose=False, step=0.01),
            np.array([1, 2, 3])
        )

    def test_predict_different_inputs(self):
        ojanet = algorithms.Oja(minimized_data_size=1, verbose=False,
                                step=0.01)

        data = np.array([[1, 2, 3]]).T
        target = np.array([[1, 2, 3]]).T

        ojanet.train(data, epsilon=0.01, epochs=100)
        self.assertInvalidVectorPred(ojanet, data.ravel(), target,
                                     decimal=2)

    def test_reconstruct_different_inputs(self):
        ojanet = algorithms.Oja(minimized_data_size=1, verbose=False,
                                step=0.01)

        data = np.array([[1, 2, 3]]).T
        target = np.array([[1, 2, 3]]).T
        input_vector = data.ravel()

        ojanet.train(data, epsilon=0.01, epochs=100)

        test_vectors = vectors_for_testing(input_vector)

        for i, test_vector in enumerate(test_vectors, start=1):
            np.testing.assert_array_almost_equal(
                ojanet.reconstruct(test_vector),
                target,
                decimal=1
            )
