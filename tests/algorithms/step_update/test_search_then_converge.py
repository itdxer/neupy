from neupy import algorithms, layers

from data import xor_input_train, xor_target_train
from base import BaseTestCase


class LearningRateUpdatesTestCase(BaseTestCase):
    def test_search_then_converge(self):
        network = algorithms.GradientDescent(
            [
                layers.Input(2),
                layers.Tanh(3),
                layers.Tanh(1),
            ],
            step=0.3,
            reduction_freq=50,
            rate_coefitient=0.2,
            addons=[algorithms.SearchThenConverge]
        )
        network.train(xor_input_train, xor_target_train, epochs=6)
        self.assertAlmostEqual(
            network.variables.step.get_value(),
            0.18,
            places=5,
        )
