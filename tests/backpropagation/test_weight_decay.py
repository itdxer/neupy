import numpy as np

from neupy import algorithms, layers

from data import xor_input_train, xor_target_train
from base import BaseTestCase


class WeightDecayTestCase(BaseTestCase):
    def test_with_bp(self):
        network = algorithms.Backpropagation(
            [
                layers.Tanh(
                    input_size=2,
                    weight=np.array([
                        [-0.04144865, -0.60458235,  0.25735483],
                        [0.08818209, -0.10212516, -1.46030816]
                    ]),
                    bias=np.array([-0.53980522, -0.64724144, -0.92496063])
                ),
                layers.Tanh(
                    input_size=3,
                    weight=np.array([
                        [0.1393251],
                        [1.59479241],
                        [0.1479949],
                    ]),
                    bias=np.array([0.54230442])
                ),
                layers.StepOutput(1, output_bounds=(-1, 1))
            ],
            step=0.3,
            decay_rate=0.0001,
            optimizations=[algorithms.WeightDecay]
        )
        network.train(xor_input_train, xor_target_train, epochs=500)
        self.assertAlmostEqual(network.last_error(), 0, places=2)
