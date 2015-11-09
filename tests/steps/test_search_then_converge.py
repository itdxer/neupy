import numpy as np

from neupy import algorithms, layers

from data import xor_input_train, xor_target_train
from base import BaseTestCase


class LearningRateUpdatesTestCase(BaseTestCase):
    def setUp(self):
        super(LearningRateUpdatesTestCase, self).setUp()
        self.first_step = 0.3
        self.connection = [
            layers.Tanh(2),
            layers.Tanh(3),
            layers.StepOutput(1, output_bounds=(-1, 1))
        ]

    def test_search_then_converge(self):
        network = algorithms.Backpropagation(
            self.connection,
            step=self.first_step,
            epochs_step_minimizator=50,
            rate_coefitient=0.2,
            optimizations=[algorithms.SearchThenConverge]
        )
        network.train(xor_input_train, xor_target_train, epochs=6)
        self.assertEqual(network.step, 0.18)
