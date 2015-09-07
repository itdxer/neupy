import numpy as np

from neupy import algorithms

from base import BaseTestCase


class PerceptronTestCase(BaseTestCase):
    def test_train(self):
        input_data = np.array([[1, 0], [2, 2], [3, 3], [0, 0]])
        target_data = np.array([[1], [-1], [-1], [1]])

        network = algorithms.LMS((2, 1), step=0.5)

        network.train(input_data, target_data, epochs=200)
        predicted_result = network.predict(np.array([[4, 4], [-1, -1]]))

        self.assertEqual(network.last_error_in(), 0)
        self.assertEqual(predicted_result[0, 0], -1)
        self.assertEqual(predicted_result[1, 0], 1)
