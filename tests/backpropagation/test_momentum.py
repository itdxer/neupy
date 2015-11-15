import numpy as np

from neupy import layers, algorithms

from data import xor_input_train, xor_target_train
from base import BaseTestCase


class MomentumTestCase(BaseTestCase):
    def test_backpropagation(self):
        input_layer = layers.Tanh(2)
        hidden_layer = layers.Tanh(3)
        output = layers.StepOutput(1, output_bounds=(-1, 1))

        network2 = algorithms.Momentum(
            (input_layer > hidden_layer > output),
            step=0.1,
            momentum=0.1,
        )

        network2.train(xor_input_train, xor_target_train, epochs=300)
        self.assertAlmostEqual(0, network2.last_error(), places=2)
