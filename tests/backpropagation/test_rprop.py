import copy

import numpy as np

from neupy import algorithms
from neupy.layers import *

from data import simple_input_train, simple_target_train
from utils import compare_networks
from base import BaseTestCase


class RPROPTestCase(BaseTestCase):
    def setUp(self):
        super(RPROPTestCase, self).setUp()

        weight1 = np.array([
            [-0.58972205,  0.38090322, -0.13283831,  0.54170814,  0.86029372],
            [-0.74337568,  0.44961596,  1.55008712, -0.71292011, -0.84660862],
            [0.78160333, -0.14645822,  0.49924888, -1.49470678, -0.89980909],
            [-0.04653505,  0.06402338, -0.62104662,  0.33968261,  0.71701342]
        ])
        weight2 = np.array([
            [-0.46880316,  0.06155588],
            [-0.46980184,  0.4019776],
            [0.25394709, -0.11112981],
            [-1.19489307, -1.13308135],
            [-0.07423586, -0.71765059],
            [-2.01427485, -1.62398753]
        ])

        input_layer = SigmoidLayer(3, weight=weight1)
        hidden_layer = SigmoidLayer(5, weight=weight2)

        self.connection = input_layer > hidden_layer > OutputLayer(2)

    def test_rprop(self):
        nw = algorithms.RPROP(
            self.connection,
            minimum_step=0.001,
            maximum_step=1,
            increase_factor=1.1,
            decrease_factor=0.1,
            step=1,
            use_raw_predict_at_error=True
        )

        nw.train(simple_input_train, simple_target_train, epochs=100)
        self.assertGreater(1e-4, nw.last_error_in())

    def test_compare_bp_and_rprop(self):
        network_default_error, network_tested_error = compare_networks(
            # Test classes
            algorithms.Backpropagation,
            algorithms.RPROP,
            # Test data
            (simple_input_train, simple_target_train),
            # Network configurations
            connection=self.connection,
            step=1,
            use_raw_predict_at_error=False,
            shuffle_data=True,
            # Test configurations
            epochs=50,
            # is_comparison_plot=True
        )
        self.assertGreater(network_default_error, network_tested_error)

    def test_irpropplus(self):
        options = dict(
            minimum_step=0.001,
            maximum_step=1,
            increase_factor=1.1,
            decrease_factor=0.1,
            step=1,
            use_raw_predict_at_error=True
        )
        nw = algorithms.IRPROPPlus(
            copy.deepcopy(self.connection), **options
        )

        nw.train(simple_input_train, simple_target_train, epochs=100)
        irprop_plus_error = nw.last_error_in()
        self.assertGreater(1e-4, nw.last_error_in())

        nw = algorithms.RPROP(copy.deepcopy(self.connection), **options)

        nw.train(simple_input_train, simple_target_train, epochs=100)
        rprop_error = nw.last_error_in()
        self.assertGreater(rprop_error, irprop_plus_error)
