import numpy as np

from neupy.algorithms import GradientDescent, WeightDecay, SearchThenConverge
from neupy.layers import Sigmoid, Tanh, StepOutput, Output

from base import BaseTestCase
from data import xor_input_train, xor_target_train


class GradientDescentTestCase(BaseTestCase):
    def test_network_attrs(self):
        network = GradientDescent((2, 2, 1), verbose=False)
        network.step = 0.1
        network.bias = True
        network.error = 'mse'
        network.shuffle_data = True

        with self.assertRaises(TypeError):
            network.step = '33'

        with self.assertRaises(ValueError):
            network.error = 'not a function'

        with self.assertRaises(TypeError):
            network.shuffle_data = 1

    def test_backpropagation(self):
        network = GradientDescent(
            (Tanh(2) > Tanh(3) > StepOutput(1, output_bounds=(-1, 1))),
            step=0.3,
            verbose=False
        )
        network.train(xor_input_train, xor_target_train, epochs=1000)
        self.assertAlmostEqual(network.last_error(), 0, places=2)

    def test_optimization_validations(self):
        with self.assertRaises(ValueError):
            # Invalid optimization class
            GradientDescent((2, 3, 1), optimizations=[GradientDescent])

        with self.assertRaises(ValueError):
            # Dublicate optimization algorithms from one type
            GradientDescent(
                (2, 3, 1), optimizations=[WeightDecay, WeightDecay]
            )

        GradientDescent((2, 3, 1), optimizations=[WeightDecay], verbose=False)
        GradientDescent((2, 3, 1), optimizations=[SearchThenConverge],
                        verbose=False)
        GradientDescent(
            (2, 3, 1),
            optimizations=[WeightDecay, SearchThenConverge],
            verbose=False
        )
