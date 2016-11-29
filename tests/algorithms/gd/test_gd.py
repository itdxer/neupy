import time
from functools import partial

import numpy as np

from neupy import algorithms, layers
from neupy.helpers.logs import TerminalLogger
from neupy.algorithms.gd.base import format_error, apply_batches

from utils import compare_networks, catch_stdout
from base import BaseTestCase
from data import simple_classification


class GradientDescentTestCase(BaseTestCase):
    def test_network_attrs(self):
        network = algorithms.GradientDescent((2, 2, 1), verbose=False)
        network.step = 0.1
        network.error = 'mse'
        network.shuffle_data = True

        with self.assertRaises(TypeError):
            network.step = '33'

        with self.assertRaises(ValueError):
            network.error = 'not a function'

        with self.assertRaises(TypeError):
            network.shuffle_data = 1

    def test_gd(self):
        x_train, _, y_train, _ = simple_classification()

        network = algorithms.GradientDescent(
            layers.Input(10) > layers.Tanh(20) > layers.Tanh(1),
            step=0.3,
            verbose=False
        )
        network.train(x_train, y_train, epochs=500)
        self.assertAlmostEqual(network.errors.last(), 0.014, places=3)

    def test_addons_exceptions(self):
        with self.assertRaises(ValueError):
            # Invalid optimization class
            algorithms.GradientDescent(
                (2, 3, 1),
                addons=[algorithms.GradientDescent]
            )

        with self.assertRaises(ValueError):
            # Dublicate optimization algorithms from one type
            algorithms.GradientDescent(
                (2, 3, 1), addons=[algorithms.WeightDecay,
                                   algorithms.WeightDecay]
            )

        algorithms.GradientDescent(
            (2, 3, 1),
            addons=[algorithms.WeightDecay],
            verbose=False,
        )
        algorithms.GradientDescent(
            (2, 3, 1),
            addons=[algorithms.SearchThenConverge],
            verbose=False,
        )
        algorithms.GradientDescent(
            (2, 3, 1),
            addons=[algorithms.WeightDecay,
                    algorithms.SearchThenConverge],
            verbose=False
        )

    def test_minibatch_gd(self):
        x_train, _, y_train, _ = simple_classification()
        compare_networks(
           # Test classes
           algorithms.GradientDescent,
           partial(algorithms.MinibatchGradientDescent, batch_size=1),
           # Test data
           (x_train, y_train),
           # Network configurations
           connection=(layers.Input(10) > layers.Tanh(20) > layers.Tanh(1)),
           step=0.1,
           shuffle_data=True,
           verbose=False,
           # Test configurations
           epochs=40,
           show_comparison_plot=False
        )

    def test_gd_get_params_method(self):
        network = algorithms.GradientDescent((2, 3, 1))

        self.assertIn('connection', network.get_params(with_connection=True))
        self.assertNotIn('connection',
                         network.get_params(with_connection=False))


class GDAdditionalFunctionsTestCase(BaseTestCase):
    def test_gd_format_error_function(self):
        self.assertEqual('?', format_error(None))
        self.assertEqual('1.00000', format_error(np.array([1])))

        self.assertEqual('101', format_error(101))
        self.assertEqual('101', format_error(101.01))

        self.assertEqual('0.12000', format_error(0.12))
        self.assertEqual('0.00000', format_error(0.00000001))

    def test_gd_apply_batches_exceptions(self):
        with self.assertRaisesRegexp(ValueError, "at least one element"):
            apply_batches(function=lambda x: x, arguments=[],
                          batch_size=12, logger=None)

    def test_gd_apply_batches(self):
        def function(x):
            time.sleep(0.02)
            print()
            return 12345

        with catch_stdout() as out:
            apply_batches(
                function=function,
                arguments=[np.ones(100)],
                batch_size=10,
                logger=TerminalLogger(),
                show_progressbar=True,
                show_error_output=True
            )
            terminal_output = out.getvalue()

        self.assertIn('12345', terminal_output)
        self.assertIn('error', terminal_output)
