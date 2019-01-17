from collections import namedtuple

import numpy as np

from neupy import algorithms, layers

from utils import reproducible_network_train
from data import xor_input_train, xor_target_train
from base import BaseTestCase


class WeightEliminationTestCase(BaseTestCase):
    def test_that_alg_works(self):
        network = algorithms.GradientDescent(
            [
                layers.Input(2),
                layers.Tanh(3),
                layers.Tanh(1),
            ],
            step=0.3,
            batch_size='all',
            zero_weight=20,
            addons=[algorithms.WeightElimination]
        )
        network.train(xor_input_train, xor_target_train, epochs=350)
        self.assertAlmostEqual(network.errors.last(), 0, places=2)

    def test_weight_minimization(self):
        base_network = reproducible_network_train()
        decay_network = reproducible_network_train(
            decay_rate=0.3,
            zero_weight=5,
            addons=[algorithms.WeightElimination]
        )

        iter_networks = zip(
            base_network.layers[1:-1],
            decay_network.layers[1:-1],
        )

        for net_layer, decay_layer in iter_networks:
            self.assertGreater(
                np.linalg.norm(self.eval(net_layer.weight)),
                np.linalg.norm(self.eval(decay_layer.weight)),
            )
