import numpy as np

from neupy import algorithms, layers

from data import xor_input_train, xor_target_train
from base import BaseTestCase


class LearningRateUpdatesTestCase(BaseTestCase):
    def test_search_then_converge(self):
        network = algorithms.GradientDescent(
            [
                layers.Tanh(2),
                layers.Tanh(3),
                layers.StepOutput(1, output_bounds=(-1, 1))
            ],
            step=0.3,
            epochs_step_minimizator=50,
            rate_coefitient=0.2,
            addons=[algorithms.SearchThenConverge]
        )
        network.train(xor_input_train, xor_target_train, epochs=6)
        self.assertEqual(network.variables.step.get_value(), 0.18)
