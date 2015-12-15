import numpy as np

from neupy import algorithms

from base import BaseTestCase


class PerceptronTestCase(BaseTestCase):
    def test_train(self):
        input_data = np.array([[1, 0], [2, 2], [3, 3], [0, 0]])
        target_data = np.array([[1], [0], [0], [1]])

        network = algorithms.LMS((2, 1), step=0.2, verbose=False)

        network.train(input_data, target_data, epochs=100)
        predicted_result = network.predict(np.array([[4, 4], [0, 0]]))

        np.testing.assert_array_almost_equal(
            predicted_result,
            np.array([[0, 1]]).T
        )

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.LMS((1, 1), verbose=False),
            np.array([1, 2, 3]),
            np.array([1, 2, 3])
        )

    def test_predict_different_inputs(self):
        lmsnet = algorithms.LMS((1, 1), verbose=False)

        data = np.array([[1, 2, 3]]).T
        target = np.array([[1, 1, 1]]).T

        lmsnet.train(data, target)
        self.assertInvalidVectorPred(lmsnet, data.ravel(), target,
                                     decimal=2)
