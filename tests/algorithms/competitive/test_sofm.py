import math

import numpy as np

from neupy import algorithms
from neupy.exceptions import WeightInitializationError
from neupy.algorithms.competitive import sofm
from neupy.algorithms.competitive.neighbours import gaussian_df

from base import BaseTestCase


X = np.array([
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


def make_circle(max_samples=100):
    data = np.random.random((max_samples, 2))
    x, y = data[:, 0], data[:, 1]

    distance_from_center = ((x - 0.5) ** 2 + (y - 0.5) ** 2)
    return data[distance_from_center <= 0.5 ** 2]


class SOFMUtilsFunctionTestsCase(BaseTestCase):
    def test_sofm_gaussian_df_zero_std(self):
        actual_output = gaussian_df(np.arange(-3, 4), mean=0, std=0)
        expected_output = np.array([0, 0, 0, 1, 0, 0, 0])
        np.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_sofm_gaussian_df(self):
        actual_output = gaussian_df(np.arange(-3, 4), mean=0, std=1)
        expected_output = np.array([
            0.23873659, 0.52907781, 0.8528642,
            1., 0.8528642, 0.52907781, 0.23873659
        ])
        np.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_sofm_decay_function(self):
        actual_output = sofm.decay_function(12, 10, reduction_rate=10)
        self.assertEqual(6, actual_output)

        actual_output = sofm.decay_function(12, 20, reduction_rate=10)
        self.assertEqual(4, actual_output)

        actual_output = sofm.decay_function(12, 30, reduction_rate=10)
        self.assertEqual(3, actual_output)


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
            np.array([[1, 0.926, 0.802, 0.956]]),
            decimal=3)


class SOFMNeigboursTestCase(BaseTestCase):
    def test_sofm_neightbours_exceptions(self):
        with self.assertRaisesRegexp(ValueError, "Cannot find center"):
            sofm.find_neighbours_on_rect_grid(
                grid=np.zeros((3, 3)),
                center=(0, 0, 0),
                radius=1)

        with self.assertRaisesRegexp(ValueError, "Cannot find center"):
            sofm.find_step_scaler_on_rect_grid(
                grid=np.zeros((3, 3)),
                center=(0, 0, 0),
                std=1)

    def test_neightbours_in_10d(self):
        actual_result = sofm.find_neighbours_on_rect_grid(
            np.zeros([3] * 10),
            center=[1] * 10,
            radius=0)
        self.assertEqual(np.sum(actual_result), 1)

    def test_neightbours_in_3d(self):
        actual_result = sofm.find_neighbours_on_rect_grid(
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
        actual_result = sofm.find_neighbours_on_rect_grid(
            np.zeros((3, 3)),
            center=(0, 0),
            radius=1)

        expected_result = np.array([
            [1., 1., 0.],
            [1., 0., 0.],
            [0., 0., 0.]
        ])
        np.testing.assert_array_equal(actual_result, expected_result)

        actual_result = sofm.find_neighbours_on_rect_grid(
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

        actual_result = sofm.find_neighbours_on_rect_grid(
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
        actual_result = sofm.find_neighbours_on_rect_grid(
            np.zeros(5),
            center=(2,),
            radius=1)

        expected_result = np.array([0, 1, 1, 1, 0])
        np.testing.assert_array_equal(actual_result, expected_result)

    def test_sofm_gaussian_neighbour_2d(self):
        expected_result = np.array([
            [0.52907781, 0.69097101, 0.7645389, 0.69097101, 0.52907781],
            [0.69097101, 0.85286420, 0.9264321, 0.85286420, 0.69097101],
            [0.76453890, 0.92643210, 1.0000000, 0.92643210, 0.76453890],
            [0.69097101, 0.85286420, 0.9264321, 0.85286420, 0.69097101],
            [0.52907781, 0.69097101, 0.7645389, 0.69097101, 0.52907781],
        ])

        actual_result = sofm.find_step_scaler_on_rect_grid(
            grid=np.zeros((5, 5)),
            center=(2, 2),
            std=1)

        np.testing.assert_array_almost_equal(
            expected_result, actual_result)

    def test_sofm_gaussian_neighbour_3d(self):
        expected_result = np.array([
            [
                [0.85286420, 0.90190947, 0.85286420],
                [0.90190947, 0.95095473, 0.90190947],
                [0.85286420, 0.90190947, 0.85286420],
            ], [
                [0.90190947, 0.95095473, 0.90190947],
                [0.95095473, 1.00000000, 0.95095473],
                [0.90190947, 0.95095473, 0.90190947],
            ], [
                [0.85286420, 0.90190947, 0.85286420],
                [0.90190947, 0.95095473, 0.90190947],
                [0.85286420, 0.90190947, 0.85286420],
            ]
        ])
        actual_result = sofm.find_step_scaler_on_rect_grid(
            grid=np.zeros((3, 3, 3)),
            center=(1, 1, 1),
            std=1)

        np.testing.assert_array_almost_equal(
            expected_result, actual_result)

    def test_sofm_hexagon_grid_neighbours(self):
        # radius 1 (odd row index)
        expected_result = np.array([
            [0, 0, 0, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 2, 1, 2, 0],
            [0, 2, 2, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        actual_result = sofm.find_neighbours_on_hexagon_grid(
            grid=np.zeros((5, 5)),
            center=(2, 2),
            radius=1)

        np.testing.assert_array_almost_equal(
            expected_result, actual_result)

        # radius 1 (even row index)
        expected_result = np.array([
            [0, 0, 2, 2, 0],
            [0, 2, 1, 2, 0],
            [0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        actual_result = sofm.find_neighbours_on_hexagon_grid(
            grid=np.zeros((5, 5)),
            center=(1, 2),
            radius=1)

        np.testing.assert_array_almost_equal(
            expected_result, actual_result)

        # radius 1 (partialy broken)
        expected_result = np.array([
            [1, 2, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        actual_result = sofm.find_neighbours_on_hexagon_grid(
            grid=np.zeros((5, 5)),
            center=(0, 0),
            radius=1)

        np.testing.assert_array_almost_equal(
            expected_result, actual_result)

        # radius 2 (partialy broken)
        expected_result = np.array([
            [0, 0, 0, 3, 2],
            [0, 0, 3, 2, 1],
            [0, 0, 0, 3, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 0, 0],
        ])
        actual_result = sofm.find_neighbours_on_hexagon_grid(
            grid=np.zeros((5, 5)),
            center=(1, 4),
            radius=2)

        np.testing.assert_array_almost_equal(
            expected_result, actual_result)

        # radius 2
        expected_result = np.array([
            [0, 3, 3, 3, 0],
            [3, 2, 2, 3, 0],
            [3, 2, 1, 2, 3],
            [3, 2, 2, 3, 0],
            [0, 3, 3, 3, 0],
        ])
        actual_result = sofm.find_neighbours_on_hexagon_grid(
            grid=np.zeros((5, 5)),
            center=(2, 2),
            radius=2)

        np.testing.assert_array_almost_equal(
            expected_result, actual_result)

        # radius 3
        expected_result = np.array([
            [0, 0, 4, 4, 4, 4, 0],
            [0, 4, 3, 3, 3, 4, 0],
            [0, 4, 3, 2, 2, 3, 4],
            [4, 3, 2, 1, 2, 3, 4],
            [0, 4, 3, 2, 2, 3, 4],
            [0, 4, 3, 3, 3, 4, 0],
            [0, 0, 4, 4, 4, 4, 0],
        ])
        actual_result = sofm.find_neighbours_on_hexagon_grid(
            grid=np.zeros((7, 7)),
            center=(3, 3),
            radius=3)

        np.testing.assert_array_almost_equal(
            expected_result, actual_result)


class SOFMTestCase(BaseTestCase):
    def setUp(self):
        super(SOFMTestCase, self).setUp()
        self.weight = np.array([
            [0.65091234, -0.52271686, 0.56344712],
            [-0.13191953, 2.43582716, -0.19703619]
        ])

    def test_invalid_attrs(self):
        with self.assertRaisesRegexp(ValueError, "Feature grid"):
            # Invalid feature grid shape
            algorithms.SOFM(n_inputs=2, n_outputs=4, features_grid=(2, 3))

        with self.assertRaisesRegexp(ValueError, "n_outputs, features_grid"):
            algorithms.SOFM(n_inputs=2)

        with self.assertRaisesRegexp(ValueError, "more than 2 dimensions"):
            sofm = algorithms.SOFM(n_inputs=2, n_outputs=3, weight=self.weight)
            sofm.train(np.zeros((10, 2, 1)))

        with self.assertRaisesRegexp(ValueError, "more than 2 dimensions"):
            sofm = algorithms.SOFM(n_inputs=2, n_outputs=3, weight=self.weight)
            sofm.predict(np.zeros((10, 2, 1)))

        with self.assertRaisesRegexp(ValueError, "Input data expected"):
            sofm = algorithms.SOFM(n_inputs=2, n_outputs=3, weight=self.weight)
            sofm.train(np.zeros((10, 10)))

        with self.assertRaisesRegexp(ValueError, "Input data expected"):
            sofm = algorithms.SOFM(n_inputs=2, n_outputs=3, weight=self.weight)
            sofm.predict(np.zeros((10, 10)))

        with self.assertRaisesRegexp(ValueError, "one or two dimensional"):
            algorithms.SOFM(n_inputs=2, features_grid=(3, 1, 1),
                            grid_type='hexagon')

    def test_sofm_1d_vector_input(self):
        sofm = algorithms.SOFM(
            n_inputs=2,
            n_outputs=3,
            weight=self.weight,
        )
        output = sofm.predict(X[0])
        self.assertEqual(output.shape, (1, 3))

    def test_sofm(self):
        sn = algorithms.SOFM(
            n_inputs=2,
            n_outputs=3,
            weight=X[(2, 0, 4), :].T,
            learning_radius=0,
            features_grid=(3,),
            shuffle_data=True,
            verbose=False,
            reduce_radius_after=None,
            reduce_step_after=None,
            reduce_std_after=None,
        )
        sn.train(X, epochs=100)

        np.testing.assert_array_almost_equal(
            sn.predict(X), answers)

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

        sn.train(X, epochs=10)

        answers = np.array([
            [0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0., 0.],
            [1., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 1., 0.],
        ])

        np.testing.assert_array_almost_equal(
            sn.predict(X),
            answers
        )

    def test_sofm_training_with_4d_grid(self):
        sofm = algorithms.SOFM(
            n_inputs=4,
            n_outputs=8,
            features_grid=(2, 2, 2),
            verbose=False,
        )

        data = np.concatenate([X, X], axis=1)

        sofm.train(data, epochs=1)
        error_after_first_epoch = sofm.training_errors[-1]

        sofm.train(data, epochs=9)
        self.assertLess(sofm.training_errors[-1], error_after_first_epoch)

    def test_sofm_angle_distance(self):
        sn = algorithms.SOFM(
            n_inputs=2,
            n_outputs=3,
            distance='cos',
            learning_radius=1,
            features_grid=(3, 1),
            weight=X[(0, 2, 4), :].T,
            verbose=False
        )
        sn.train(X, epochs=3)

        answers = np.array([
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 1.],
        ])
        np.testing.assert_array_almost_equal(
            sn.predict(X),
            answers)

    def test_sofm_weight_norm_after_training_with_custom_weights(self):
        sofm = algorithms.SOFM(
            n_inputs=2,
            n_outputs=3,
            distance='cos',
            learning_radius=1,
            features_grid=(3, 1),
            weight=X[(0, 2, 4), :].T,
            verbose=False
        )

        actual_norms = np.linalg.norm(sofm.weight, axis=0)
        expected_norms = np.array([1, 1, 1])
        np.testing.assert_array_almost_equal(expected_norms, actual_norms)

        sofm.train(X, epochs=6)

        actual_norms = np.linalg.norm(sofm.weight, axis=0)
        expected_norms = np.array([1, 1, 1])
        np.testing.assert_array_almost_equal(expected_norms, actual_norms)

    def test_sofm_weight_norm_before_training(self):
        sofm = algorithms.SOFM(
            n_inputs=2,
            n_outputs=3,
            verbose=False,
            distance='cos',
        )

        actual_norms = np.linalg.norm(sofm.weight, axis=0)
        expected_norms = np.array([1, 1, 1])

        np.testing.assert_array_almost_equal(expected_norms, actual_norms)

    def test_train_different_inputs(self):
        self.assertInvalidVectorTrain(
            algorithms.SOFM(n_inputs=1, n_outputs=1, verbose=False),
            X.ravel())

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

        sofmnet.train(X.ravel())
        self.assertInvalidVectorPred(sofmnet, X.ravel(),
                                     target, decimal=2)

    def test_sofm_std_parameter(self):
        default_params = dict(
            n_inputs=2,
            n_outputs=3,
            learning_radius=1,
            weight=self.weight)

        sofm_1 = algorithms.SOFM(std=1, **default_params)
        sofm_1.train(X[:1], epochs=1)
        dist_1 = np.linalg.norm(sofm_1.weight[0, :] - sofm_1.weight[1, :])

        sofm_0 = algorithms.SOFM(std=0.1, **default_params)
        sofm_0.train(X[:1], epochs=1)
        dist_0 = np.linalg.norm(sofm_0.weight[0, :] - sofm_0.weight[1, :])

        # Since SOFM-1 has bigger std than SOFM-0, two updated
        # neurons should be closer to each other for SOFM-1 than
        # for SOFM-0
        self.assertLess(dist_1, dist_0)

    def test_sofm_n_outputs_as_optional_parameter(self):
        sofm = algorithms.SOFM(
            n_inputs=2,
            features_grid=(10, 2, 3),
        )
        self.assertEqual(60, sofm.n_outputs)

    def test_sofm_hexagon_grid(self):
        data = make_circle(max_samples=100)
        sofm = algorithms.SOFM(
            n_inputs=2,
            n_outputs=9,
            learning_radius=1,
            reduce_radius_after=4,
            features_grid=(3, 3),
            verbose=False,
            grid_type='hexagon',
        )
        sofm.train(data, epochs=10)
        grid = sofm.weight.reshape((2, 3, 3))

        center = grid[:, 1, 1]
        top_left = grid[:, 0, 0]
        top_right = grid[:, 0, 2]

        distance_top_left = np.linalg.norm(center - top_left)
        distance_top_right = np.linalg.norm(center - top_right)

        self.assertLess(distance_top_right, distance_top_left)

        bottom_left = grid[:, 2, 0]
        bottom_right = grid[:, 2, 2]

        distance_bottom_left = np.linalg.norm(center - bottom_left)
        distance_bottom_right = np.linalg.norm(center - bottom_right)

        self.assertLess(distance_bottom_right, distance_bottom_left)

    def test_sofm_storage(self):
        X = np.random.random((100, 10))
        sofm = algorithms.SOFM(
            n_inputs=10,
            features_grid=(10, 10),
            std=1,
            step=0.5,
            learning_radius=2,
            weight=np.random.random((10, 100))
        )

        sofm.train(X, epochs=10)
        self.assertPickledNetwork(sofm, X)

        parameters = sofm.get_params()
        self.assertIn('weight', parameters)
        self.assertIsInstance(parameters['weight'], np.ndarray)


class SOFMParameterReductionTestCase(BaseTestCase):
    def test_sofm_step_reduction(self):
        X = 4 * np.ones((1, 2))
        default_params = dict(
            n_inputs=2,
            n_outputs=1,
            step=0.1,
            weight=np.ones((2, 1)))

        sofm_1 = algorithms.SOFM(reduce_step_after=1, **default_params)
        sofm_1.train(X, epochs=4)
        dist_1 = np.linalg.norm(sofm_1.weight.T - X)

        sofm_2 = algorithms.SOFM(reduce_step_after=10, **default_params)
        sofm_2.train(X, epochs=4)
        dist_2 = np.linalg.norm(sofm_2.weight.T - X)

        # SOFM-2 suppose to be closer, becase learning rate decreases
        # slower with bigger reduction factor
        self.assertLess(dist_2, dist_1)

        sofm_3 = algorithms.SOFM(reduce_step_after=None, **default_params)
        sofm_3.train(X, epochs=4)
        dist_3 = np.linalg.norm(sofm_3.weight.T - X)

        # SOFM-3 doesn't have any step reduction, so it
        # converges even faster than SOFM-2
        self.assertLess(dist_3, dist_2)

    def test_sofm_std_reduction(self):
        X = 4 * np.ones((1, 2))
        default_params = dict(
            n_inputs=2,
            n_outputs=3,
            weight=np.ones((2, 3)),

            std=1,

            learning_radius=1,
            reduce_radius_after=None)

        sofm_1 = algorithms.SOFM(reduce_std_after=1, **default_params)
        sofm_1.train(X, epochs=4)
        dist_1 = np.linalg.norm(sofm_1.weight.T - X)

        sofm_2 = algorithms.SOFM(reduce_std_after=10, **default_params)
        sofm_2.train(X, epochs=4)
        dist_2 = np.linalg.norm(sofm_2.weight.T - X)

        # SOFM-2 suppose to be closer, becase std decreases
        # slower with bigger reduction factor
        self.assertLess(dist_2, dist_1)

        sofm_3 = algorithms.SOFM(reduce_std_after=None, **default_params)
        sofm_3.train(X, epochs=4)
        dist_3 = np.linalg.norm(sofm_3.weight.T - X)

        # SOFM-3 doesn't have any std reduction, so it
        # converges even faster than SOFM-2
        self.assertLess(dist_3, dist_2)

    def test_sofm_learning_radius_reduction(self):
        X = 4 * np.ones((1, 2))
        default_params = dict(
            n_inputs=2,
            n_outputs=3,
            weight=np.ones((2, 3)),
            learning_radius=1)

        sofm_1 = algorithms.SOFM(reduce_radius_after=1, **default_params)
        sofm_1.train(X, epochs=4)
        dist_1 = np.linalg.norm(sofm_1.weight.T - X)

        sofm_2 = algorithms.SOFM(reduce_radius_after=3, **default_params)
        sofm_2.train(X, epochs=4)
        dist_2 = np.linalg.norm(sofm_2.weight.T - X)

        # SOFM-2 suppose to be closer, becase learning radius
        # decreases slower with bigger reduction factor
        self.assertLess(dist_2, dist_1)

        sofm_3 = algorithms.SOFM(reduce_radius_after=None, **default_params)
        sofm_3.train(X, epochs=4)
        dist_3 = np.linalg.norm(sofm_3.weight.T - X)

        # SOFM-3 doesn't have any learning radius reduction, so it
        # converges even faster than SOFM-2
        self.assertLess(dist_3, dist_2)

    def test_sofm_custom_parameter_reduction(self):
        X = 4 * np.ones((1, 2))
        default_params = dict(
            n_inputs=2,
            n_outputs=3,
            weight=np.ones((2, 3)),

            std=1,
            step=0.1,
            learning_radius=1,

            reduce_radius_after=None,
            reduce_step_after=None,
            reduce_std_after=None)

        sofm_1 = algorithms.SOFM(**default_params)
        sofm_1.train(X, epochs=4)
        dist_1 = np.linalg.norm(sofm_1.weight.T - X)

        def on_epoch_end_update_radius(network):
            if network.last_epoch % 2 == 0:
                network.learning_radius = 0
            else:
                network.learning_radius = 1

        def on_epoch_end_update_step(network):
            network.step = 0.1 / network.last_epoch

        def on_epoch_end_update_std(network):
            network.std = 1. / network.last_epoch

        testcases = {
            'learning_radius': on_epoch_end_update_radius,
            'step': on_epoch_end_update_step,
            'std': on_epoch_end_update_std,
        }

        for testcase_name, on_epoch_end in testcases.items():
            sofm_2 = algorithms.SOFM(
                epoch_end_signal=on_epoch_end,
                **default_params
            )
            sofm_2.train(X, epochs=4)
            dist_2 = np.linalg.norm(sofm_2.weight.T - X)

            self.assertLess(dist_1, dist_2,
                            msg="Test case name: {}".format(testcase_name))


class SOFMWeightInitializationTestCase(BaseTestCase):
    def test_sofm_weight_init_exceptions(self):
        msg = "Cannot apply PCA"
        with self.assertRaisesRegexp(WeightInitializationError, msg):
            algorithms.SOFM(
                n_inputs=4,
                n_outputs=7,
                weight='init_pca',
                grid_type='hexagon'
            )

    def test_sample_data_function(self):
        X = np.random.random((10, 4))
        sampled_weights = sofm.sample_data(X, features_grid=(7, 1))
        self.assertEqual(sampled_weights.shape, (4, 7))

        X = np.random.random((3, 4))
        sampled_weights = sofm.sample_data(X, features_grid=(7, 1))
        self.assertEqual(sampled_weights.shape, (4, 7))

    def test_sofm_init_during_the_training(self):
        sofm = algorithms.SOFM(
            n_inputs=4,
            n_outputs=7,
            weight='sample_from_data',
        )

        X = np.random.random((10, 4))
        self.assertTrue(callable(sofm.weight))

        sofm.train(X, epochs=1)
        self.assertFalse(callable(sofm.weight))

    def test_sample_data_weight_init_in_sofm(self):
        sofm = algorithms.SOFM(
            n_inputs=4,
            n_outputs=7,
            weight='sample_from_data',
        )

        X = np.random.random((10, 4))
        self.assertTrue(callable(sofm.weight))

        sofm.init_weights(X)
        self.assertFalse(callable(sofm.weight))
        self.assertEqual(sofm.weight.shape, (4, 7))

    def test_linear_weight_init_in_sofm(self):
        sofm = algorithms.SOFM(
            n_inputs=4,
            features_grid=(3, 3),
            weight='init_pca',
        )

        X = np.random.random((100, 4))
        self.assertTrue(callable(sofm.weight))

        sofm.init_weights(X)
        self.assertFalse(callable(sofm.weight))
        self.assertEqual(sofm.weight.shape, (4, 9))

        for row in (0, 3, 6):
            left = sofm.weight[:, row]
            center = sofm.weight[:, row + 1]
            right = sofm.weight[:, row + 2]

            self.assertLess(
                np.linalg.norm((left - center) ** 2),
                np.linalg.norm((left - right) ** 2))

        for i in range(3):
            top = sofm.weight[:, i]
            center = sofm.weight[:, i + 3]
            bottom = sofm.weight[:, i + 6]

            self.assertLess(
                np.linalg.norm((top - center) ** 2),
                np.linalg.norm((top - bottom) ** 2))

    def test_sofm_double_initialization_exception_cos_distance(self):
        sofm = algorithms.SOFM(
            n_inputs=2,
            n_outputs=3,
            verbose=False,
            distance='cos',
            weight='sample_from_data'
        )
        sofm.init_weights(X)

        with self.assertRaises(WeightInitializationError):
            sofm.init_weights(X)

    def test_sofm_double_initialization_exception(self):
        sofm = algorithms.SOFM(
            n_inputs=2,
            n_outputs=3,
            verbose=False,
            weight='sample_from_data'
        )
        sofm.init_weights(X)

        with self.assertRaises(WeightInitializationError):
            sofm.init_weights(X)
