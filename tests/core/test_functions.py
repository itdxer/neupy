from functools import partial

import numpy as np

from neupy.functions import with_derivative
from neupy.functions import *
from neupy.functions import *

from base import BaseTestCase


def differentiate(func, args, epsilon=1e-5):
    args = np.array(args)
    return (func(args + epsilon) - func(args - epsilon)) / (2 * epsilon)


class FunctionTestCase(BaseTestCase):
    def test_derivatives(self):
        def constant(x):
            return 2

        @with_derivative(constant)
        def linear(x):
            return 2 * x

        @with_derivative(linear)
        def square(x):
            return x ** 2

        # Check output
        self.assertEqual(square(2), 4)
        self.assertEqual(linear(1), 2)
        self.assertEqual(constant(100), 2)

        # Check derivative
        self.assertEqual(square.deriv(1), 2)
        self.assertEqual(square.deriv.deriv(100), 2)

    def test_error_functions(self):
        data1 = np.array([[3], [0], [-2]])
        data2 = np.array([[0], [0], [0]])

        self.assertEqual(mse(data1, data2), 13 / 3.)
        square_error_deriv = mse.deriv(data1, data2)
        np.testing.assert_array_equal(square_error_deriv,
                               np.array([[6], [0], [-4]]) / 3.)

        data1 = np.array([[0.99], [0.01], [0.5]])
        data2 = np.array([[0.5], [0.5], [0.5]])
        self.assertEqual(
            np.round(cross_entropy_error(data1, data2), 4), np.array([1.7695])
        )

        self.assertEqual(
            np.round(
                kullback_leibler(np.array([[0.1]]), np.array([[0.2]])), 4
            ),
            0.0444
        )

    def test_step(self):
        self.assertEqual(step(-1), 0)
        self.assertEqual(step(0), 0)
        self.assertEqual(step(1), 1)

        test_matrix = np.array([
            [-1, -100, 0.0001, -0.0001],
            [0, 100, 1, 1.0001],
        ])
        np.testing.assert_array_almost_equal(
            step(test_matrix),
            np.array([
                [0, 0, 1, 0],
                [0, 1, 1, 1],
            ])
        )

    def test_linear(self):
        self.assertEqual(linear(-1), -1)
        self.assertEqual(linear.deriv(100), 1)

        test_matrix = np.random.random((4, 5))
        np.testing.assert_array_almost_equal(
            linear(test_matrix),
            test_matrix
        )
        np.testing.assert_array_almost_equal(
            linear.deriv(test_matrix),
            np.ones(test_matrix.shape)
        )

    def test_sigmoid_scalars(self):
        self.assertAlmostEqual(sigmoid(0.7), 0.668, places=3)
        self.assertAlmostEqual(sigmoid(0.7, alpha=-0.1), 0.483, places=3)

        self.assertAlmostEqual(
            sigmoid.deriv(0.7),
            differentiate(sigmoid, 0.7)
        )
        self.assertAlmostEqual(
            sigmoid.deriv(0.7, alpha=-0.1),
            differentiate(partial(sigmoid, alpha=-0.1), 0.7)
        )
        self.assertAlmostEqual(
            sigmoid.deriv.deriv(0.7),
            differentiate(sigmoid.deriv, 0.7)
        )
        self.assertAlmostEqual(
            sigmoid.deriv.deriv(0.7, alpha=2),
            differentiate(partial(sigmoid.deriv, alpha=2), 0.7)
        )

    def test_sigmoid_vectors(self):
        test_vector = np.array([1, 2, 4, -0.33])

        np.testing.assert_array_almost_equal(
            sigmoid(test_vector),
            np.array([0.731,  0.881,  0.982,  0.418]),
            decimal=3
        )
        np.testing.assert_array_almost_equal(
            sigmoid.deriv.deriv(test_vector, alpha=2),
            differentiate(partial(sigmoid.deriv, alpha=2), test_vector)
        )
        np.testing.assert_array_almost_equal(
            sigmoid.deriv(test_vector, alpha=2),
            differentiate(partial(sigmoid, alpha=2), test_vector)
        )

    def test_sigmoid_matrix(self):
        test_matrix = np.array([
            [1, 2, 4, -0.33],
            [-10, 0, 10, 1.45],
        ])

        np.testing.assert_array_almost_equal(
            sigmoid(test_matrix),
            np.array([
                [0.731, 0.881, 0.982, 0.418],
                [0, 0.5, 1, 0.81],
            ]),
            decimal=3
        )
        np.testing.assert_array_almost_equal(
            sigmoid.deriv.deriv(test_matrix, alpha=2),
            differentiate(partial(sigmoid.deriv, alpha=2), test_matrix)
        )
        np.testing.assert_array_almost_equal(
            sigmoid.deriv(test_matrix, alpha=2),
            differentiate(partial(sigmoid, alpha=2), test_matrix)
        )

    def test_tanh_scalar(self):
        self.assertAlmostEqual(tanh(0.7), 0.604, places=3)
        self.assertAlmostEqual(tanh(0.7, 0.5), 0.336, places=3)

        self.assertAlmostEqual(
            tanh.deriv(0.7),
            differentiate(tanh, 0.7)
        )
        self.assertAlmostEqual(
            tanh.deriv(0.7, alpha=-0.1),
            differentiate(partial(tanh, alpha=-0.1), 0.7)
        )
        self.assertAlmostEqual(
            tanh.deriv.deriv(0.7),
            differentiate(tanh.deriv, 0.7)
        )
        self.assertAlmostEqual(
            tanh.deriv.deriv(0.7, alpha=2),
            differentiate(partial(tanh.deriv, alpha=2), 0.7)
        )

    def test_tanh_vectors(self):
        test_vector = np.array([1, 2, 4, -0.33])

        np.testing.assert_array_almost_equal(
            tanh(test_vector),
            np.array([0.762, 0.964, 0.999, -0.319]),
            decimal=3
        )
        np.testing.assert_array_almost_equal(
            tanh.deriv.deriv(test_vector, alpha=2),
            differentiate(partial(tanh.deriv, alpha=2), test_vector)
        )
        np.testing.assert_array_almost_equal(
            tanh.deriv(test_vector, alpha=2),
            differentiate(partial(tanh, alpha=2), test_vector)
        )

    def test_tanh_matrix(self):
        test_matrix = np.array([
            [1, 2, 4, -0.33],
            [-10, 0, 10, 1.45],
        ])

        np.testing.assert_array_almost_equal(
            tanh(test_matrix),
            np.array([
                [0.762, 0.964, 0.999, -0.319],
                [-1, 0, 1, 0.896],
            ]),
            decimal=3
        )
        np.testing.assert_array_almost_equal(
            tanh.deriv.deriv(test_matrix, alpha=2),
            differentiate(partial(tanh.deriv, alpha=2), test_matrix)
        )
        np.testing.assert_array_almost_equal(
            tanh.deriv(test_matrix, alpha=2),
            differentiate(partial(tanh, alpha=2), test_matrix)
        )

    def test_rectifier(self):
        self.assertEqual(rectifier(1), 1)
        self.assertEqual(rectifier(0), 0)
        self.assertEqual(rectifier(-1), 0)

        np.testing.assert_array_almost_equal(
            rectifier(
                np.array([
                    [-10, -0.33, 0, -1],
                    [10, 0, 1, 0.33],
                ])
            ),
            np.array([
                [0, 0, 0, 0],
                [10, 0, 1, 0.33],
            ])
        )

    def test_softplus_scalar(self):
        self.assertAlmostEqual(softplus(1), 1.313, places=3)
        self.assertAlmostEqual(softplus(0), 0.693, places=3)
        self.assertAlmostEqual(softplus(-2), 0.127, places=3)

        self.assertAlmostEqual(
            softplus.deriv(0.7),
            differentiate(softplus, 0.7)
        )
        self.assertAlmostEqual(
            softplus.deriv.deriv(0.7),
            differentiate(softplus.deriv, 0.7)
        )

    def test_softplus_vectors(self):
        test_vector = np.array([1, 2, 4, -0.33])

        np.testing.assert_array_almost_equal(
            softplus(test_vector),
            np.array([1.313, 2.127, 4.018, 0.542]),
            decimal=3
        )
        np.testing.assert_array_almost_equal(
            softplus.deriv.deriv(test_vector),
            differentiate(softplus.deriv, test_vector)
        )
        np.testing.assert_array_almost_equal(
            softplus.deriv(test_vector),
            differentiate(softplus, test_vector)
        )

    def test_softplus_matrix(self):
        test_matrix = np.array([
            [1, 2, 4, -0.33],
            [-10, 0, 10, 1.45],
        ])

        np.testing.assert_array_almost_equal(
            softplus(test_matrix),
            np.array([
                [1.313, 2.127, 4.018, 0.542],
                [0, 0.693, 10, 1.661],
            ]),
            decimal=3
        )
        np.testing.assert_array_almost_equal(
            softplus.deriv.deriv(test_matrix),
            differentiate(softplus.deriv, test_matrix)
        )
        np.testing.assert_array_almost_equal(
            softplus.deriv(test_matrix),
            differentiate(softplus, test_matrix)
        )

    def test_softmax(self):
        input_matrix = np.random.random((4, 5))
        result = softmax(input_matrix)
        self.assertEqual(result.sum(), 4)

        np.testing.assert_array_almost_equal(
            softmax(np.array([[1, 2, 3]])),
            np.array([[0.09,  0.2447, 0.6652]]),
            decimal=4
        )
        np.testing.assert_array_almost_equal(
            softmax.deriv(np.array([[4, 2, 3]])),
            np.array([
                [[0.223, -0.06, -0.163]],
                [[-0.06, 0.082, -0.022]],
                [[-0.163, -0.022, 0.185]],
            ]),
            decimal=3
        )
