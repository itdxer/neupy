import numpy as np

from neupy.algorithms import (Backpropagation, WeightDecay,
                              LeakStepAdaptation)
from neupy.layers import (Sigmoid, Tanh, StepOutput,
                          Output)

from base import BaseTestCase
from data import xor_input_train, xor_target_train


class BackpropagationTestCase(BaseTestCase):
    def test_network_attrs(self):
        network = Backpropagation((2, 2, 1), verbose=False)
        network.step = 0.1
        network.bias = True
        network.error = 'mse'
        network.shuffle_data = True

        with self.assertRaises(TypeError):
            network.step = '33'

        with self.assertRaises(TypeError):
            network.use_bias = 123

        with self.assertRaises(ValueError):
            network.error = 'not a function'

        with self.assertRaises(TypeError):
            network.shuffle_data = 1

    def test_backpropagation(self):
        output = StepOutput(1, output_bounds=(-1, 1))

        bias1 = np.array([0.31319847, -1.17858149, 0.71556407])
        weight1 = np.array([
            [1.60798015, 0.16304449, -0.22483005],
            [-0.90144173, 0.58500625, -0.01724167]
        ])
        bias2 = np.array([-1.34351428])
        weight2 = np.array([
            [0.45506056],
            [0.24790366],
            [-0.74360389]
        ])

        input_layer = Tanh(2, weight=weight1, bias=bias1)
        hidden_layer = Tanh(3, weight=weight2, bias=bias2)

        network = Backpropagation(
            (input_layer > hidden_layer > output),
            step=0.3,
            verbose=False
        )

        network.train(xor_input_train, xor_target_train, epochs=1000)
        self.assertAlmostEqual(network.last_error(), 0, places=2)

    def test_optimization_validations(self):
        with self.assertRaises(ValueError):
            # Invalid optimization class
            Backpropagation((2, 3, 1), optimizations=[Backpropagation])

        with self.assertRaises(ValueError):
            # Dublicate optimization algorithms from one type
            Backpropagation(
                (2, 3, 1), optimizations=[WeightDecay, WeightDecay]
            )

        Backpropagation((2, 3, 1), optimizations=[WeightDecay], verbose=False)
        Backpropagation((2, 3, 1), optimizations=[LeakStepAdaptation],
                        verbose=False)
        Backpropagation(
            (2, 3, 1),
            optimizations=[WeightDecay, LeakStepAdaptation],
            verbose=False
        )
