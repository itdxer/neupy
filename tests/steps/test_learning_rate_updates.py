import numpy as np

from neupy.algorithms import *
from neupy.layers import TanhLayer, StepOutputLayer

from data import xor_input_train, xor_target_train
from base import BaseTestCase


class LearningRateUpdatesTestCase(BaseTestCase):
    def setUp(self):
        super(LearningRateUpdatesTestCase, self).setUp()
        self.first_step = 0.3
        # Weights
        self.weight1 = np.array([
            [0.57030714, 0.64724479, 0.74482306],
            [0.12310346, 0.26571213, 0.74472318],
            [0.5642351, 0.52127089, 0.57070108],
        ])
        self.weight2 = np.array([
            [0.2343891],
            [0.70945912],
            [0.46677056],
            [0.83986252],
        ])
        # Layers
        input_layer = TanhLayer(2, weight=self.weight1)
        hidden_layer = TanhLayer(3, weight=self.weight2)
        output = StepOutputLayer(1, output_bounds=(-1, 1))
        self.connection = input_layer > hidden_layer > output

    def test_simple_learning_rate_minimization(self):
        network = Backpropagation(
            self.connection,
            step=self.first_step,
            epochs_step_minimizator=50,
            use_raw_predict_at_error=True,
            optimizations=[SimpleStepMinimization]
        )
        network.train(xor_input_train, xor_target_train, epochs=101)
        self.assertEqual(network.step, self.first_step / 3)

    def test_quadration_learning_rate_minimiation(self):
        network = Backpropagation(
            self.connection,
            step=self.first_step,
            epochs_step_minimizator=50,
            rate_coefitient=0.2,
            optimizations=[SearchThenConverge]
        )
        network.train(xor_input_train, xor_target_train, epochs=7)
        self.assertEqual(network.step, 0.18)

    def test_error_difference_update(self):
        network = Backpropagation(
            self.connection,
            step=self.first_step,
            update_for_smaller_error=1.05,
            update_for_bigger_error=0.7,
            error_difference=1.04,
            use_raw_predict_at_error=True,
            optimizations=[ErrorDifferenceStepUpdate]
        )
        network.train(xor_input_train, xor_target_train, epochs=200)
        self.assertEqual(round(network.last_error_in(), 5), 0)
