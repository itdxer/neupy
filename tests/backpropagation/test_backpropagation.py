import numpy as np

from neupy.algorithms import (Backpropagation, WeightDecay,
                                 LeakStepAdaptation)
from neupy.layers import (SigmoidLayer, TanhLayer, StepOutputLayer,
                             OutputLayer)
from neupy.functions import with_derivative

from base import BaseTestCase
from data import xor_input_train, xor_target_train


class BackpropagationTestCase(BaseTestCase):
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
            use_raw_predict_at_error=True,
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

        weight1_new = np.array([[0.50461013, 0.50437699],
                                [0.50461013, 0.50437699]])
        weight2_new = np.array([[0.53691945, 0.53781823]]).T

        input_layer = SigmoidLayer(2, weight=weight1)
        hidden_layer = SigmoidLayer(2, weight=weight2)
        output = OutputLayer(1)

        network = Backpropagation(
            (input_layer > hidden_layer > output),
            error=square_error,
            step=1,
        )

        network.train(np.array([[1, 1]]), np.array([[1]]), epochs=1)

        trained_weight1 = network.train_layers[0].weight_without_bias
        trained_weight2 = network.train_layers[1].weight_without_bias

        self.assertTrue(np.all(
            np.round(trained_weight1, 8) == np.round(weight1_new, 8))
        )
        self.assertTrue(np.all(
            np.round(trained_weight2, 8) == np.round(weight2_new, 8))
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

        Backpropagation((2, 3, 1), optimizations=[WeightDecay])
        Backpropagation((2, 3, 1), optimizations=[LeakStepAdaptation])
        Backpropagation(
            (2, 3, 1),
            optimizations=[WeightDecay, LeakStepAdaptation]
        )
