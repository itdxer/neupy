import numpy as np

from neupy.algorithms import (Backpropagation, WeightDecay,
                              LeakStepAdaptation)
from neupy.layers import (SigmoidLayer, TanhLayer, StepOutputLayer,
                          OutputLayer)
from neupy.functions import with_derivative

from base import BaseTestCase
from data import xor_input_train, xor_target_train


class BackpropagationTestCase(BaseTestCase):
    def test_network_attrs(self):
        network = Backpropagation((2, 2, 1), verbose=False)
        network.step = 0.1
        network.bias = True
        network.error = lambda x: x
        network.shuffle_data = True

        with self.assertRaises(TypeError):
            network.step = '33'

        with self.assertRaises(TypeError):
            network.use_bias = 123

        with self.assertRaises(TypeError):
            network.error = 'not a function'

        with self.assertRaises(TypeError):
            network.shuffle_data = 1

    def test_backpropagation(self):
        output = StepOutputLayer(1, output_bounds=(-1, 1))

        weight1 = np.array([
            [0.31319847, -1.17858149, 0.71556407],
            [1.60798015, 0.16304449, -0.22483005],
            [-0.90144173, 0.58500625, -0.01724167]
        ])
        weight2 = np.array([
            [-1.34351428],
            [0.45506056],
            [0.24790366],
            [-0.74360389]
        ])

        input_layer = TanhLayer(2, weight=weight1)
        hidden_layer = TanhLayer(3, weight=weight2)

        network = Backpropagation(
            (input_layer > hidden_layer > output),
            step=0.3,
            verbose=False
        )

        network.train(xor_input_train, xor_target_train, epochs=1000)
        self.assertEqual(round(network.last_error_in(), 2), 0)

    def test_first_step_updates(self):
        def square_error_deriv(output_train, target_train):
            return output_train - target_train

        @with_derivative(square_error_deriv)
        def square_error(output_train, target_train):
            return np.sum((target_train - output_train) ** 2) / 2

        weight1 = np.array([[0.1, 0.2], [0.5, 0.5], [0.5, 0.5]])
        weight2 = np.array([[0.3, 0.5, 0.5]]).T

        input_layer = SigmoidLayer(2, weight=weight1)
        hidden_layer = SigmoidLayer(2, weight=weight2)
        output = OutputLayer(1)

        network = Backpropagation(
            (input_layer > hidden_layer > output),
            error=square_error,
            step=1,
            verbose=False
        )

        test_input = np.array([[1, 1]])
        test_target = np.array([[1]])
        network.train(test_input, test_target, epochs=1)

        np.testing.assert_array_almost_equal(
            network.train_layers[0].weight_without_bias,
            np.array([[0.50461013, 0.50437699],
                      [0.50461013, 0.50437699]]),
        )
        np.testing.assert_array_almost_equal(
            network.train_layers[1].weight_without_bias,
            np.array([[0.53691945, 0.53781823]]).T,
        )

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
