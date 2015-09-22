from functools import partial

import numpy as np

from neupy.algorithms import LeakStepAdaptation, Backpropagation
from neupy.layers import SigmoidLayer, StepOutputLayer

from base import BaseTestCase
from data import even_input_train, even_target_train
from utils import compare_networks


class LeakStepAdaptationTestCase(BaseTestCase):
    def setUp(self):
        super(LeakStepAdaptationTestCase, self).setUp()

        weight1 = np.array([
            [-1.82990278, -0.21861533, -0.10817557],
            [0.00418764, -0.20416605,  0.62476191],
            [0.91992406,  0.46878743, -1.90503238]
        ])
        weight2 = np.array([
            [-1.27068127],
            [0.10575739],
            [0.27213559],
            [-0.69731429]
        ])

        input_layer = SigmoidLayer(2, weight=weight1)
        hidden_layer = SigmoidLayer(3, weight=weight2)

        self.connection = input_layer > hidden_layer > StepOutputLayer(1)

    def test_adaptive_learning_rate(self):
        network_default_error, network_tested_error = compare_networks(
            # Test classes
            Backpropagation,
            partial(
                Backpropagation,
                # Adaptive learning rate settings
                leak_size=0.5,
                alpha=0.5,
                beta=0.5,
                optimizations=[LeakStepAdaptation]
            ),
            # Test data
            (even_input_train, even_target_train),
            # Network configurations
            connection=self.connection,
            step=0.1,
            use_raw_predict_at_error=True,
            shuffle_data=True,
            # Adaptive learning rate parameters
            # Test configurations
            epochs=30,
            # is_comparison_plot=True,
        )
        self.assertGreater(network_default_error, network_tested_error)
