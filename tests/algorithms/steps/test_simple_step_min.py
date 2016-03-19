import numpy as np

from neupy import algorithms, layers
from neupy.utils import asfloat

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

    def test_simple_learning_rate_minimization(self):
        network = algorithms.GradientDescent(
            self.connection,
            step=self.first_step,
            epochs_step_minimizator=50,
            addons=[algorithms.SimpleStepMinimization]
        )
        network.train(xor_input_train, xor_target_train, epochs=100)
        self.assertAlmostEqual(
            network.variables.step.get_value(),
            asfloat(self.first_step / 3),
            places=5,
        )
