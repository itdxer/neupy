import numpy as np
import theano.tensor as T

from neupy import layers, init
from neupy.utils import asfloat

from base import BaseTestCase


class ConnectionCompilationTestCase(BaseTestCase):
    def test_simple_connection_compilation(self):
        input_matrix = asfloat(np.ones((7, 10)))
        expected_output = np.ones((7, 5))

        network = layers.join(
            layers.Input(10),
            layers.Linear(5, weight=init.Constant(0.1), bias=None)
        )

        # Generated input variables
        predict = network.compile()
        actual_output = predict(input_matrix)
        np.testing.assert_array_almost_equal(actual_output, expected_output)

        # Pre-defined input variables
        input_variable = T.matrix('x')
        predict = network.compile(input_variable)
        actual_output = predict(input_matrix)
        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_compilation_multiple_inputs(self):
        input_matrix = asfloat(np.ones((7, 10)))
        expected_output = np.ones((7, 5))

        network = layers.join(
            [[
                layers.Input(10),
            ], [
                layers.Input(10),
            ]],
            layers.Elementwise(),
            layers.Linear(5, weight=init.Constant(0.1), bias=None)
        )

        # Generated input variables
        predict = network.compile()
        actual_output = predict(input_matrix * 0.7, input_matrix * 0.3)
        np.testing.assert_array_almost_equal(actual_output, expected_output)

        # Pre-defined input variables
        input_variable_1 = T.matrix('x1')
        input_variable_2 = T.matrix('x2')

        predict = network.compile(input_variable_1, input_variable_2)
        actual_output = predict(input_matrix * 0.7, input_matrix * 0.3)
        np.testing.assert_array_almost_equal(actual_output, expected_output)

    def test_compilation_exceptions(self):
        network = [layers.Input(2), layers.Input(2)] > layers.Concatenate()
        with self.assertRaises(ValueError):
            # n_input_vars != n_input_layers
            network.compile(T.matrix('x'), T.matrix('y'), T.matrix('z'))

    def test_compilation_multiple_outputs(self):
        input_matrix = asfloat(np.ones((7, 10)))
        expected_output_1 = np.ones((7, 5))
        expected_output_2 = np.ones((7, 2))

        network = layers.join(
            layers.Input(10),
            [[
                layers.Linear(5, weight=init.Constant(0.1), bias=None)
            ], [
                layers.Linear(2, weight=init.Constant(0.1), bias=None)
            ]]
        )
        predict = network.compile()

        actual_output_1, actual_output_2 = predict(input_matrix)

        np.testing.assert_array_almost_equal(
            actual_output_1, expected_output_1)

        np.testing.assert_array_almost_equal(
            actual_output_2, expected_output_2)
