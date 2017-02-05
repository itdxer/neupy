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

        iter_networks = zip(base_network.layers[1:-1],
                            decay_network.layers[1:-1])

        for net_layer, decay_layer in iter_networks:
            self.assertGreater(
                np.linalg.norm(net_layer.weight.get_value()),
                np.linalg.norm(decay_layer.weight.get_value()),
            )

    def test_with_step_minimization_alg(self):
        default_step = 0.3
        net1 = reproducible_network_train(step=default_step)
        net2 = reproducible_network_train(
            step=default_step,
            decay_rate=0.25,
            zero_weight=10,
            addons=[algorithms.WeightElimination]
        )
        net3 = reproducible_network_train(
            step=default_step,
            decay_rate=0.25,
            zero_weight=10,
            addons=[algorithms.WeightElimination,
                    algorithms.StepDecay]
        )

        # Check that step is valid for each network
        StepCase = namedtuple('StepCase', 'network expected_step')
        step_test_cases = (
            StepCase(network=net1, expected_step=default_step),
            StepCase(network=net2, expected_step=default_step),
            StepCase(network=net3, expected_step=default_step / 6.),
        )

        for case in step_test_cases:
            step = case.network.variables.step
            self.assertAlmostEqual(step.get_value(), case.expected_step,
                                   places=2)

        # Compare weight norm between networks
        WeightNormCase = namedtuple('WeightNormCase',
                                    'with_smaller_norm with_bigger_norm')
        norm_test_cases = (
            WeightNormCase(with_smaller_norm=net2, with_bigger_norm=net1),
            WeightNormCase(with_smaller_norm=net3, with_bigger_norm=net1),
        )
        for case in norm_test_cases:
            network_layers = zip(case.with_smaller_norm.layers[1:-1],
                                 case.with_bigger_norm.layers[1:-1])
            for smaller_norm, bigger_norm in network_layers:
                weight_smaller_norm = smaller_norm.weight.get_value()
                weight_bigger_norm = bigger_norm.weight.get_value()
                self.assertGreater(
                    np.linalg.norm(weight_bigger_norm),
                    np.linalg.norm(weight_smaller_norm)
                )
