import numpy as np

from neupy import algorithms, layers

from utils import reproducible_network_train
from data import xor_input_train, xor_target_train
from base import BaseTestCase


class WeightEliminationTestCase(BaseTestCase):
    def test_that_alg_works(self):
        network = algorithms.Backpropagation(
            [
                layers.Tanh(2),
                layers.Tanh(3),
                layers.StepOutput(1, output_bounds=(-1, 1))
            ],
            step=0.3,
            zero_weight=20,
            optimizations=[algorithms.WeightElimination]
        )
        network.train(xor_input_train, xor_target_train, epochs=350)
        self.assertAlmostEqual(network.last_error(), 0, places=2)

    def test_weight_minimization(self):
        base_network = reproducible_network_train()
        decay_network = reproducible_network_train(
            decay_rate=0.3,
            zero_weight=5,
            optimizations=[algorithms.WeightElimination]
        )

        iter_networks = zip(base_network.train_layers,
                            decay_network.train_layers)

        for net_layer, decay_layer in iter_networks:
            print(
                np.linalg.norm(net_layer.weight.get_value()),
                np.linalg.norm(decay_layer.weight.get_value()),
            )

        decay_network.plot_errors()
