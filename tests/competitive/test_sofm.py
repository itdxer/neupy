import numpy as np

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
            weight=self.weight,
            learning_radius=0,
            features_grid=(3, 1),
            verbose=False
        )

        sn.train(input_data, epochs=100)
        np.testing.assert_array_almost_equal(
            sn.predict(input_data),
            answers
        )

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
            transform='euclid',
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

    def test_sofm_angle_distance(self):
        sn = algorithms.SOFM(
            n_inputs=2,
            n_outputs=3,
            transform='cos',
            learning_radius=1,
            features_grid=(3, 1),
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
            answers
        )

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.SOFM(n_inputs=1, n_outputs=1, verbose=False),
            input_data.ravel()
        )

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
