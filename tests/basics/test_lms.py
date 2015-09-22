import numpy as np

from neupy import algorithms

from base import BaseTestCase


class PerceptronTestCase(BaseTestCase):
    def test_train(self):
        input_data = np.array([[1, 0], [2, 2], [3, 3], [0, 0]])
        target_data = np.array([[1], [0], [0], [1]])

        network = algorithms.LMS((2, 1), step=0.2)

        network.train(input_data, target_data, epochs=100)
        predicted_result = network.predict(np.array([[4, 4], [0, 0]]))

        np.testing.assert_array_almost_equal(
            predicted_result,
            np.array([[0, 1]]).T
        )
