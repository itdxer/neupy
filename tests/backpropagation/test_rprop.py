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
        self.connection = Sigmoid(3) > Sigmoid(10) > Output(2)

    def test_rprop(self):
        nw = algorithms.RPROP(
            self.connection,
            minimum_step=0.001,
            maximum_step=1,
            increase_factor=1.1,
            decrease_factor=0.1,
            step=1,
            verbose=False
        )

        nw.train(simple_input_train, simple_target_train, epochs=100)
        self.assertGreater(1e-4, nw.last_error())

    def test_compare_bp_and_rprop(self):
        compare_networks(
            # Test classes
            algorithms.Backpropagation,
            algorithms.RPROP,
            # Test data
            (simple_input_train, simple_target_train),
            # Network configurations
            connection=self.connection,
            step=1,
            shuffle_data=True,
            verbose=False,
            # Test configurations
            epochs=50,
            # is_comparison_plot=True
        )

    def test_irpropplus(self):
        options = dict(
            minimum_step=0.001,
            maximum_step=1,
            increase_factor=1.1,
            decrease_factor=0.1,
            step=1,
            verbose=False
        )
        nw = algorithms.IRPROPPlus(copy.deepcopy(self.connection), **options)

        nw.train(simple_input_train, simple_target_train, epochs=100)
        irprop_plus_error = nw.last_error()
        self.assertGreater(1e-4, nw.last_error())

        nw = algorithms.RPROP(copy.deepcopy(self.connection), **options)

        nw.train(simple_input_train, simple_target_train, epochs=100)
        rprop_error = nw.last_error()
        self.assertGreater(rprop_error, irprop_plus_error)

    def test_rprop_exceptions(self):
        raise NotImplementedError()
