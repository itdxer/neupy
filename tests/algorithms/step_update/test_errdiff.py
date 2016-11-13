from neupy import algorithms, layers

from data import xor_input_train, xor_target_train
from base import BaseTestCase


class LearningRateUpdatesTestCase(BaseTestCase):
    def test_errdiff(self):
        network = algorithms.GradientDescent(
            [
                layers.Input(2),
                layers.Tanh(3),
                layers.Tanh(1),
            ],
            step=0.3,
            update_for_smaller_error=1.05,
            update_for_bigger_error=0.7,
            error_difference=1.04,
            addons=[algorithms.ErrDiffStepUpdate]
        )
        network.train(xor_input_train, xor_target_train, epochs=200)
        self.assertAlmostEqual(network.errors.last(), 0, places=5)
