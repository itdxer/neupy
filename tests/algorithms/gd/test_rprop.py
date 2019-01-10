import copy
from functools import partial

import numpy as np

from neupy import algorithms, init, layers
from neupy.layers import Input, Sigmoid
from neupy.utils import asfloat

from helpers import compare_networks
from base import BaseTestCase


simple_x_train = asfloat(np.array([
    [0.1, 0.1, 0.2],
    [0.2, 0.3, 0.4],
    [0.1, 0.7, 0.2],
]))
simple_y_train = asfloat(np.array([
    [0.2, 0.2],
    [0.3, 0.3],
    [0.5, 0.5],
]))


class RPROPTestCase(BaseTestCase):
    def setUp(self):
        super(RPROPTestCase, self).setUp()
        self.network = Input(3) > Sigmoid(10) > Sigmoid(2)

    def test_rprop(self):
        nw = algorithms.RPROP(
            self.network,
            minstep=0.001,
            maxstep=1,
            increase_factor=1.1,
            decrease_factor=0.1,
            step=1,
            verbose=False
        )

        nw.train(simple_x_train, simple_y_train, epochs=100)
        self.assertGreater(1e-4, nw.training_errors[-1])

    def test_compare_bp_and_rprop(self):
        compare_networks(
            # Test classes
            partial(algorithms.GradientDescent, batch_size=None),
            partial(algorithms.RPROP, maxstep=0.1),
            # Test data
            (simple_x_train, simple_y_train),
            # Network configurations
            network=self.network,
            step=0.1,
            shuffle_data=True,
            verbose=False,
            # Test configurations
            epochs=50,
            show_comparison_plot=False
        )

    def test_irpropplus(self):
        options = dict(
            minstep=0.001,
            maxstep=1,
            increase_factor=1.1,
            decrease_factor=0.1,
            step=1,
            verbose=False
        )

        uniform = init.Uniform()
        params1 = dict(
            weight=uniform.sample((3, 10), return_array=True),
            bias=uniform.sample((10,), return_array=True),
        )
        params2 = dict(
            weight=uniform.sample((10, 2), return_array=True),
            bias=uniform.sample((2,), return_array=True),
        )

        network = layers.join(
            Input(3),
            Sigmoid(10, **params1),
            Sigmoid(2, **params2),
        )

        nw = algorithms.IRPROPPlus(copy.deepcopy(network), **options)
        nw.train(simple_x_train, simple_y_train, epochs=100)
        irprop_plus_error = nw.training_errors[-1]
        self.assertGreater(1e-4, nw.training_errors[-1])

        nw = algorithms.RPROP(copy.deepcopy(network), **options)
        nw.train(simple_x_train, simple_y_train, epochs=100)
        rprop_error = nw.training_errors[-1]
        self.assertGreater(rprop_error, irprop_plus_error)

    def test_rprop_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.RPROP,
                minstep=1e-5,
                step=0.05,
                maxstep=1.0,

                increase_factor=1.5,
                decrease_factor=0.5,

                verbose=False,
                show_epoch=100,
            ),
            epochs=5000,
            min_accepted_error=0.006,
        )

    def test_irproplus_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.IRPROPPlus,
                minstep=1e-5,
                step=0.05,
                maxstep=1.0,

                increase_factor=1.5,
                decrease_factor=0.5,

                verbose=False,
                show_epoch=100,
            ),
            epochs=5000,
            min_accepted_error=0.005,
        )
