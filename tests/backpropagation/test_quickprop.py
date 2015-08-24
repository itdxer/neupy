from functools import partial

import numpy as np

from neuralpy import algorithms, layers

from data import simple_input_train, simple_target_train
from utils import compare_networks
from base import BaseTestCase


class QuickPropTestCase(BaseTestCase):
    def setUp(self):
        super(QuickPropTestCase, self).setUp()

        weight1 = np.array([
            [-0.3262846, -0.3899363, -1.31438701, -0.43736622,  0.1234716],
            [-0.31548075, -0.66254391,  0.78722273, -0.51545504, -0.51205823],
            [-0.38036544,  0.34930878,  1.20590571,  0.55030264, -0.94516753],
            [-2.05032326, -0.10582341, -0.33530722,  0.74043659, -0.74645546]
        ])
        weight2 = np.array([
            [-0.25706768,  0.2581464],
            [0.43860057, -0.16620158],
            [-0.87493652,  0.58832669],
            [-1.17300652, -0.21716063],
            [0.66715383, -1.46908589],
            [-1.23662587, -0.85808783]
        ])

        input_layer = layers.SigmoidLayer(3, weight=weight1)
        hidden_layer = layers.SigmoidLayer(5, weight=weight2)
        output_layer = layers.OutputLayer(2)

        self.connection = input_layer > hidden_layer > output_layer

    def test_quickprop(self):
        qp = algorithms.Quickprop(
            self.connection,
            step=0.1,
            upper_bound=1,
            use_raw_predict_at_error=False,
            shuffle_data=False,
        )
        qp.train(simple_input_train, simple_target_train, epochs=100)
        result = np.round(qp.predict(simple_input_train), 3)
        norm = np.linalg.norm(result - simple_target_train)
        self.assertGreater(1e-2, norm)

    def test_compare_quickprop_and_bp(self):
        network_default_error, network_tested_error = compare_networks(
            # Test classes
            algorithms.Backpropagation,
            partial(algorithms.Quickprop, upper_bound=1),
            # Test data
            (simple_input_train, simple_target_train),
            # Network configurations
            connection=self.connection,
            step=0.1,
            use_raw_predict_at_error=False,
            shuffle_data=False,
            # Test configurations
            epochs=100,
            # is_comparison_plot=True
        )
        self.assertGreater(network_default_error, network_tested_error)
