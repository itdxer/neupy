from neupy import algorithms, layers
from neupy.utils import asfloat

from data import xor_input_train, xor_target_train
from base import BaseTestCase


class LearningRateUpdatesTestCase(BaseTestCase):
    def test_simple_learning_rate_minimization(self):
        first_step = 0.3
        network = algorithms.GradientDescent(
            [
                layers.Input(2),
                layers.Tanh(3),
                layers.Tanh(1),
            ],
            step=first_step,
            reduction_freq=50,
            addons=[algorithms.StepDecay]
        )
        network.train(xor_input_train, xor_target_train, epochs=100)
        self.assertAlmostEqual(
            network.variables.step.get_value(),
            asfloat(first_step / 3),
            places=5,
        )
