import numpy as np

from neuralpy.layers import TanhLayer, StepOutputLayer
from neuralpy.algorithms import WeightElimination, Backpropagation

from data import xor_input_train, xor_target_train
from base import BaseTestCase


class WeightEliminationTestCase(BaseTestCase):
    def test_backpropagation(self):
        weight1 = np.array([
            [0.22667075,  0.38116981,  0.62686969],
            [1.13062085,  0.40836474, -0.50492125],
            [-0.22645265,  1.13541005, -2.7876409]
        ])
        weight2 = np.array([
            [0.63547163],
            [0.63347214],
            [-1.3669957],
            [-0.42770718]
        ])

        input_layer = TanhLayer(2, weight=weight1)
        hidden_layer = TanhLayer(3, weight=weight2)
        output = StepOutputLayer(1, output_bounds=(-1, 1))

        network = Backpropagation(
            input_layer > hidden_layer > output,
            step=0.3,
            zero_weight=20,
            use_raw_predict_at_error=True,
            optimizations=[WeightElimination]
        )
        network.train(xor_input_train, xor_target_train, epochs=350)
        self.assertEqual(round(network.last_error_in(), 2), 0)
