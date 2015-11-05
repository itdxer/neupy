import numpy as np

from neupy.layers import Tanh, StepOutput
from neupy.algorithms import Momentum

from data import xor_input_train, xor_target_train
from base import BaseTestCase


class MomentumTestCase(BaseTestCase):
    def test_backpropagation(self):
        weight1 = np.array([
            [-0.53980522, -0.64724144, -0.92496063],
            [-0.04144865, -0.60458235,  0.25735483],
            [0.08818209, -0.10212516, -1.46030816]
        ])
        weight2 = np.array([
            [0.54230442],
            [0.1393251],
            [1.59479241],
            [0.1479949]
        ])

        input_layer = Tanh(2, weight=weight1)
        hidden_layer = Tanh(3, weight=weight2)
        output = StepOutput(1, output_bounds=(-1, 1))

        network2 = Momentum(
            (input_layer > hidden_layer > output),
            step=0.1,
            momentum=0.1,
        )

        network2.train(xor_input_train, xor_target_train, epochs=300)
        self.assertEqual(round(network2.last_error(), 2), 0)
