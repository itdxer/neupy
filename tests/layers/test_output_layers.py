import numpy as np

from neupy.layers.connections import NetworkConnectionError
from neupy.layers import *

from base import BaseTestCase


class OutputLayersOperationsTestCase(BaseTestCase):
    def test_error_handling(self):
        layer = Output(1)

        with self.assertRaises(NetworkConnectionError):
            layer.relate_to(Output(1))

    def test_output_layer(self):
        layer = Output(1)
        input_vector = np.array([1, 1000, -0.1])
        output_vector = layer.output(input_vector)
        np.testing.assert_array_equal(input_vector, output_vector)

    def test_rounded_output_layer(self):
        input_vector = np.array([[1.1, 1.5, -1.99, 2]]).T

        layer = RoundedOutput(1)
        output_vector = layer.output(input_vector)
        np.testing.assert_array_equal(
            np.array([[1, 2, -2, 2]]).T,
            output_vector
        )

        layer = RoundedOutput(1, decimals=1)
        output_vector = layer.output(input_vector)
        np.testing.assert_array_equal(
            np.array([[1.1, 1.5, -2, 2]]).T,
            output_vector
        )

    def test_step_output_layer(self):
        input_vector = np.array([[-10, 0, 10, 0.001]]).T

        layer = StepOutput(1)
        output_vector = layer.output(input_vector)
        np.testing.assert_array_equal(
            np.array([[0, 0, 1, 1]]).T,
            output_vector
        )

        layer = StepOutput(1, output_bounds=(-1, 1))
        output_vector = layer.output(input_vector)
        np.testing.assert_array_equal(
            np.array([[-1, -1, 1, 1]]).T,
            output_vector
        )

        layer = StepOutput(1, critical_point=0.1)
        output_vector = layer.output(input_vector)
        np.testing.assert_array_equal(
            np.array([[0, 0, 1, 0]]).T,
            output_vector
        )

    def test_competitive_output_layer(self):
        layer = CompetitiveOutput(1)
        input_vector = np.array([[1, 10, 20, 0, -10]])
        output_vector = layer.output(input_vector)

        np.testing.assert_array_equal(
            np.array([[0, 0, 1, 0, 0]]),
            output_vector
        )

    def test_argmax_output_layer(self):
        layer = ArgmaxOutput(5)
        input_matrix = np.array([
            [1., 4, 2, 3, -10],
            [-10, 1, 0, 3.0001, 3],
            [0, 0, 0, 0, 0],
        ])
        output_vector = layer.output(input_matrix)

        np.testing.assert_array_equal(
            np.array([1, 3, 0]),
            output_vector
        )
