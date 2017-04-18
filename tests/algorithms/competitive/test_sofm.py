import math

import numpy as np

from neupy import algorithms, init
from neupy.algorithms.competitive import sofm

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


class SOFMDistanceFunctionsTestCase(BaseTestCase):
    def assert_invalid_distance_function(self, func, vector, weight,
                                         expected, decimal=6):
        np.testing.assert_array_almost_equal(
            func(vector, weight),
            expected,
            decimal=decimal)

    def test_euclid_transform(self):
        self.assert_invalid_distance_function(
            sofm.neg_euclid_distance,
            np.array([[1, 2, 3]]),
            np.array([
                [1, 2, 3],
                [1, 1, 1],
                [0, 0, 1],
                [0, 1, 2],
            ]).T,
            np.array([[0, -math.sqrt(5), -3, -math.sqrt(3)]])
        )

    def test_cosine_transform(self):
        self.assert_invalid_distance_function(
            sofm.cosine_similarity,
            np.array([[1, 2, 3]]),
            np.array([
                [1, 2, 3],
                [1, 1, 1],
                [0, 0, 1],
                [0, 1, 2],
            ]).T,
            np.array([[1,  0.926, 0.802, 0.956]]),
            decimal=3)


class SOFMNeigboursTestCase(BaseTestCase):
    def test_sofm_neightbours_exceptions(self):
        with self.assertRaisesRegexp(ValueError, "Cannot find center"):
            sofm.neuron_neighbours(
                neurons=np.zeros((3, 3)),
                center=(0, 0, 0),
                radius=1)

    def test_neightbours_in_10d(self):
        actual_result = sofm.neuron_neighbours(
            np.zeros([3] * 10),
            center=[1] * 10,
            radius=0)
        self.assertEqual(np.sum(actual_result), 1)

    def test_neightbours_in_3d(self):
        actual_result = sofm.neuron_neighbours(
            np.zeros((5, 5, 3)),
            center=(2, 2, 1),
            radius=2)

        expected_result = np.array([[
            [0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 1., 1., 1., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0.]
        ], [
            [0., 0., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 0., 0.]
        ], [
            [0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 1., 1., 1., 0.],
            [0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0.]
        ]])
        expected_result = np.transpose(expected_result, (1, 2, 0))
        np.testing.assert_array_equal(actual_result, expected_result)

    def test_neightbours_in_2d(self):
        actual_result = sofm.neuron_neighbours(
            np.zeros((3, 3)),
            center=(0, 0),
            radius=1)

        expected_result = np.array([
            [1., 1., 0.],
            [1., 0., 0.],
            [0., 0., 0.]
        ])
        np.testing.assert_array_equal(actual_result, expected_result)

        actual_result = sofm.neuron_neighbours(
            np.zeros((5, 5)),
            center=(2, 2),
            radius=2)

        expected_result = np.array([
            [0., 0., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 0., 0.]
        ])
        np.testing.assert_array_equal(actual_result, expected_result)

        actual_result = sofm.neuron_neighbours(
            np.zeros((3, 3)),
            center=(1, 1),
            radius=0)

        expected_result = np.array([
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.]
        ])
        np.testing.assert_array_equal(actual_result, expected_result)

    def test_neightbours_in_1d(self):
        actual_result = sofm.neuron_neighbours(
            np.zeros(5),
            center=(2,),
            radius=1)
        expected_result = np.array([0, 1, 1, 1, 0])
        np.testing.assert_array_equal(actual_result, expected_result)


class SOFMTestCase(BaseTestCase):
    def setUp(self):
        super(SOFMTestCase, self).setUp()
        self.weight = np.array([
            [0.65091234, -0.52271686, 0.56344712],
            [-0.13191953, 2.43582716, -0.19703619]
        ])

    def test_invalid_attrs(self):
        with self.assertRaises(ValueError):
            # Invalid feature grid shape
            algorithms.SOFM(
                n_inputs=2,
                n_outputs=4,
                learning_radius=0,
                features_grid=(2, 3),
                verbose=False
            )

    def test_sofm(self):
        sn = algorithms.SOFM(
            n_inputs=2,
            n_outputs=3,
            weight=input_data[(2, 0, 4), :].T,
            learning_radius=0,
            features_grid=(3,),
            shuffle_data=True,
            verbose=False,
            reduce_radius_after=None,
            reduce_step_after=None,
            reduce_std_after=None,
        )
        sn.train(input_data, epochs=100)

        np.testing.assert_array_almost_equal(
            sn.predict(input_data), answers)

    def test_sofm_euclide_norm_distance(self):
        weight = np.array([
            [1.41700099, 0.52680476],
            [-0.60938464, 1.56545643],
            [-0.30243644, 0.13994967],
            [-0.07456091, 0.54797268],
            [-1.12894803, 0.32702141],
            [0.92084690, 0.02683249],
        ]).T
        sn = algorithms.SOFM(
            n_inputs=2,
            n_outputs=6,
            weight=weight,
            distance='euclid',
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

        np.testing.assert_array_almost_equal(
            sn.predict(input_data),
            answers
        )

    def test_sofm_training_with_4d_grid(self):
        sofm = algorithms.SOFM(
            n_inputs=4,
            n_outputs=8,
            features_grid=(2, 2, 2),
            verbose=False,
        )

        data = np.concatenate([input_data, input_data], axis=1)

        sofm.train(data, epochs=1)
        error_after_first_epoch = sofm.errors.last()

        sofm.train(data, epochs=9)
        self.assertLess(sofm.errors.last(), error_after_first_epoch)

    def test_sofm_angle_distance(self):
        sn = algorithms.SOFM(
            n_inputs=2,
            n_outputs=3,
            distance='cos',
            learning_radius=1,
            features_grid=(3, 1),
            weight=input_data[(0, 2, 4), :].T,
            verbose=False
        )
        sn.train(input_data, epochs=6)

        answers = np.array([
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 1.],
        ])
        np.testing.assert_array_almost_equal(
            sn.predict(input_data),
            answers)

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.SOFM(n_inputs=1, n_outputs=1, verbose=False),
            input_data.ravel())

    def test_predict_different_inputs(self):
        sofmnet = algorithms.SOFM(n_inputs=1, n_outputs=2, verbose=False)
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
