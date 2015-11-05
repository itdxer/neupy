import numpy as np

from neupy.layers import (CompetitiveOutput, Linear, Output,
                             EuclideDistanceLayer, AngleDistanceLayer)
from neupy import algorithms
from neupy.algorithms.competitive.sofm import neuron_neighbours
from base import BaseTestCase


input_data = np.array([
    [0.1961, 0.9806],
    [-0.1961, 0.9806],
    [0.9806, 0.1961],
    [0.9806, -0.1961],
    [-0.5812, -0.8137],
    [-0.8137, -0.5812],
])
answers = np.array([
    [0., 1., 0.],
    [0., 1., 0.],
    [1., 0., 0.],
    [1., 0., 0.],
    [0., 0., 1.],
    [0., 0., 1.],
])


class SOFMTestCase(BaseTestCase):
    def setUp(self):
        super(SOFMTestCase, self).setUp()

        self.weight = np.array([
            [0.65091234, -0.52271686, 0.56344712],
            [-0.13191953, 2.43582716, -0.19703619]
        ])

    def test_neightbours(self):
        result = neuron_neighbours(np.zeros((3, 3)), (0, 0), 1)
        answer = np.array([
            [1., 1., 0.],
            [1., 0., 0.],
            [0., 0., 0.]
        ])
        self.assertTrue(np.all(result == answer))

        result = neuron_neighbours(np.zeros((5, 5)), (2, 2), 2)
        answer = np.array([
            [0., 0., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 0., 0.]
        ])
        self.assertTrue(np.all(result == answer))

        result = neuron_neighbours(np.zeros((3, 3)), (1, 1), 0)
        answer = np.array([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.]
        ])
        self.assertTrue(np.all(result == answer))

    def test_invalid_attrs(self):
        with self.assertRaises(ValueError):
            algorithms.SOFM(
                Linear(2) > Output(3),
                learning_radius=-1,
                verbose=False
            )

        with self.assertRaises(ValueError):
            algorithms.SOFM(
                Linear(2) > Output(2),
                learning_radius=1,
                verbose=False
            )

        with self.assertRaises(ValueError):
            algorithms.SOFM(
                Linear(2) > CompetitiveOutput(4),
                learning_radius=-1,
                features_grid=(2, 3),
                verbose=False
            )

    def test_sofm(self):
        input_layer = Linear(2, weight=self.weight)
        output_layer = CompetitiveOutput(3)

        sn = algorithms.SOFM(
            input_layer > output_layer,
            learning_radius=0,
            features_grid=(3, 1),
            verbose=False
        )

        sn.train(input_data, epochs=100)

        for data, answer in zip(input_data, answers):
            network_output = sn.predict(np.reshape(data, (2, 1)).T)
            correct_result = np.reshape(answer, (3, 1)).T
            self.assertTrue(np.all(network_output == correct_result))

    def test_sofm_euclide_norm_distance(self):
        weight = np.array([
            [1.41700099, 0.52680476],
            [-0.60938464, 1.56545643],
            [-0.30243644, 0.13994967],
            [-0.07456091, 0.54797268],
            [-1.12894803, 0.32702141],
            [0.92084690, 0.02683249],
        ]).T
        input_layer = EuclideDistanceLayer(2, weight=weight)
        output_layer = CompetitiveOutput(6)

        sn = algorithms.SOFM(
            input_layer > output_layer,
            learning_radius=1,
            features_grid=(3, 2),
            verbose=False
        )

        sn.train(input_data, epochs=10)

        answers = np.array([
            [0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0., 0.],
            [1., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1., 0.],
        ])

        for data, answer in zip(input_data, answers):
            network_output = sn.predict(np.reshape(data, (2, 1)).T)
            correct_result = np.reshape(answer, (6, 1)).T
            self.assertTrue(np.all(network_output == correct_result))

    def test_sofm_angle_distance(self):
        input_layer = AngleDistanceLayer(2, weight=self.weight)
        output_layer = CompetitiveOutput(3)

        sn = algorithms.SOFM(
            input_layer > output_layer,
            learning_radius=1,
            features_grid=(3, 1),
            verbose=False
        )

        sn.train(input_data, epochs=10)

        answers = np.array([
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 1.],
        ])

        for data, answer in zip(input_data, answers):
            network_output = sn.predict(np.reshape(data, (2, 1)).T)
            correct_result = np.reshape(answer, (3, 1)).T
            self.assertTrue(np.all(network_output == correct_result))

    def test_train_different_inputs(self):
        input_layer = Linear(1)
        output_layer = CompetitiveOutput(1)

        self.assertInvalidVectorTrain(
            algorithms.SOFM(input_layer > output_layer, verbose=False),
            input_data.ravel()
        )
    #
    def test_predict_different_inputs(self):
        input_layer = Linear(1)
        output_layer = CompetitiveOutput(2)

        sofmnet = algorithms.SOFM(input_layer > output_layer, verbose=False)
        target = np.array([
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ])

        sofmnet.train(input_data.ravel())
        self.assertInvalidVectorPred(sofmnet, input_data.ravel(), target,
                                     decimal=2)
