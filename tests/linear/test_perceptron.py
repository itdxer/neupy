import numpy as np

from neupy import algorithms, layers
from neupy.layers.connections import NetworkConnectionError

from base import BaseTestCase


class PerceptronTestCase(BaseTestCase):
    def test_perceptron_init_errors(self):
        with self.assertRaises(ValueError):
            algorithms.Perceptron((2, 2, 1), verbose=False)

        with self.assertRaises(ValueError):
            algorithms.Perceptron((2, 2.5), verbose=False)

        with self.assertRaises(NetworkConnectionError):
            algorithms.Perceptron(
                layers.Sigmoid(2) > layers.Output(1),
                verbose=False
            )

    def test_valid_cases(self):
        algorithms.Perceptron(
            layers.Step(2) > layers.Output(1),
            verbose=False
        )

    def test_train(self):
        input_data = np.array([[1, 0], [2, 2], [3, 3], [0, 0]])
        target_data = np.array([[1], [0], [0], [1]])

        prnet = algorithms.Perceptron((2, 1), step=0.1, verbose=False)

        prnet.train(input_data, target_data, epochs=30)
        predicted_result = prnet.predict(np.array([[4, 4], [0, 0]]))

        self.assertEqual(prnet.last_error(), 0)
        self.assertEqual(predicted_result[0, 0], 0)
        self.assertEqual(predicted_result[1, 0], 1)

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.Perceptron((1, 1), verbose=False),
            np.array([1, 2, 3]),
            np.array([1, 2, 3])
        )

    def test_predict_different_inputs(self):
        pnet = algorithms.Perceptron((1, 1), verbose=False)

        data = np.array([[1, 2, 3]]).T
        target = np.array([[1, 1, 1]]).T

        pnet.train(data, target)
        self.assertInvalidVectorPred(pnet, data.ravel(), target,
                                     decimal=2)
