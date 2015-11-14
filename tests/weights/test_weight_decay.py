import numpy as np

from neupy import algorithms, layers

from utils import reproducible_network_train
from data import xor_input_train, xor_target_train
from base import BaseTestCase


class WeightDecayTestCase(BaseTestCase):
    def test_that_alg_works(self):
        network = algorithms.Backpropagation(
            [
                layers.Tanh(2),
                layers.Tanh(3),
                layers.StepOutput(1, output_bounds=(-1, 1))
            ],
            step=0.3,
            decay_rate=0.0001,
            optimizations=[algorithms.WeightDecay]
        )
        network.train(xor_input_train, xor_target_train, epochs=500)
        self.assertAlmostEqual(network.last_error(), 0, places=2)

    def test_weight_minimization(self):
        base_network = reproducible_network_train()
        decay_network = reproducible_network_train(
            decay_rate=0.1,
            optimizations=[algorithms.WeightDecay]
        )

        iter_networks = zip(base_network.train_layers,
                            decay_network.train_layers)

        for net_layer, decay_layer in iter_networks:
            self.assertGreater(
                np.linalg.norm(net_layer.weight.get_value()),
                np.linalg.norm(decay_layer.weight.get_value()),
            )

    def test_with_step_minimization_alg(self):
        pass
