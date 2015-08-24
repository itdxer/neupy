import numpy as np

from neuralpy.functions import with_derivative
from neuralpy.functions import *
from neuralpy.functions import *

from base import BaseTestCase


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
        self.assertEqualArrays(square_error_deriv,
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

    def test_signum(self):
        # signum
        self.assertEqual(signum(-1), 0)
        self.assertEqual(signum(1), 1)
        self.assertEqual(signum(1, upper_value=10, lower_value=-10), 10)
        self.assertEqual(signum(-0.01, upper_value=10, lower_value=-10), -10)

    def test_linear(self):
        # linear
        self.assertEqual(linear(-1), -1)
        self.assertEqual(linear.deriv(100), 1)

    def test_sigmoid(self):
        # sigmoid
        self.assertEqual(round(sigmoid(0.7), 3), 0.668)
        self.assertEqual(round(sigmoid(0.7, alpha=-0.1), 3), 0.483)
        self.assertEqual(round(sigmoid.deriv(0.7), 3), 0.222)
        self.assertEqual(round(sigmoid.deriv(0.7, alpha=-0.1), 3), -0.025)
        self.assertEqual(round(sigmoid.deriv.deriv(0.7), 3), -0.075)
        self.assertEqual(
            round(sigmoid.deriv.deriv(0.7, alpha=2), 3), -0.384
        )

    def test_tanh(self):
        # tanh
        self.assertEqual(round(tanh(0.7), 3), 0.604)
        self.assertEqual(round(tanh(0.7, 0.5), 3), 0.336)
        self.assertEqual(round(tanh.deriv(0.7, 0.1), 3), 0.1)
        self.assertEqual(round(tanh.deriv.deriv(0.7, 0.8), 3), -0.482)

    def test_rectifier(self):
        # rectifier
        self.assertEqual(rectifier(1), 1)
        self.assertEqual(rectifier(0), 0)
        self.assertEqual(rectifier(-1), 0)

    def test_softplus(self):
        # softplus
        self.assertEqual(round(softplus(1), 5), 1.31326)
        self.assertEqual(round(softplus(0), 5), 0.69315)
        self.assertEqual(round(softplus(-2), 5), 0.12693)

    def test_softmax(self):
        # softmax
        result = softmax(np.random.random((4, 5)))
        self.assertEqual(result.sum(), 4)

        actual = softmax(np.array([[1, 2, 3]]))
        expected = np.array([[0.09,  0.2447, 0.6652]])
        self.assertEqualArrays(np.round(actual, 4), expected)
