import copy

from neupy import algorithms
from neupy.layers import Input, Sigmoid
from neupy import init

from data import simple_input_train, simple_target_train
from utils import compare_networks
from base import BaseTestCase


class RPROPTestCase(BaseTestCase):
    def setUp(self):
        super(RPROPTestCase, self).setUp()
        self.connection = Input(3) > Sigmoid(10) > Sigmoid(2)

    def test_rprop(self):
        nw = algorithms.RPROP(
            self.connection,
            minstep=0.001,
            maxstep=1,
            increase_factor=1.1,
            decrease_factor=0.1,
            step=1,
            verbose=False
        )

        nw.train(simple_input_train, simple_target_train, epochs=100)
        self.assertGreater(1e-4, nw.errors.last())

    def test_compare_bp_and_rprop(self):
        compare_networks(
            # Test classes
            algorithms.GradientDescent,
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
        connection = [
            Input(3),
            Sigmoid(10, weight=init.Uniform(), bias=init.Uniform()),
            Sigmoid(2, weight=init.Uniform(), bias=init.Uniform()),
        ]

        nw = algorithms.IRPROPPlus(copy.deepcopy(connection), **options)
        nw.train(simple_input_train, simple_target_train, epochs=100)
        irprop_plus_error = nw.errors.last()
        self.assertGreater(1e-4, nw.errors.last())

        nw = algorithms.RPROP(copy.deepcopy(connection), **options)
        nw.train(simple_input_train, simple_target_train, epochs=100)
        rprop_error = nw.errors.last()
        self.assertGreater(rprop_error, irprop_plus_error)

    def test_rprop_exceptions(self):
        test_algorithms = [
            algorithms.RPROP,
            algorithms.IRPROPPlus
        ]

        for algorithm_class in test_algorithms:
            with self.assertRaises(ValueError):
                algorithm_class(self.connection,
                                addons=[algorithms.ErrDiffStepUpdate])

            # But this code should work fine
            algorithm_class(self.connection,
                            addons=[algorithms.WeightDecay])
