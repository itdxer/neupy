import textwrap
from collections import namedtuple

import numpy as np
from sklearn import datasets
from neupy import algorithms, layers
from neupy.exceptions import StopTraining
from neupy.algorithms.base import (ErrorHistoryList, show_network_options,
                                   logging_info_about_the_data,
                                   parse_show_epoch_property)

from utils import catch_stdout
from base import BaseTestCase


xor_zero_input_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_zero_target_train = np.array([[1, 0, 0, 1]]).T


class NetworkMainTestCase(BaseTestCase):
    def test_training_epoch(self):
        data, target = datasets.make_classification(30, n_features=10,
                                                    n_classes=2)
        network = algorithms.GradientDescent((10, 3, 1))

        self.assertEqual(network.last_epoch, 0)

        network.train(data, target, epochs=10)
        self.assertEqual(network.last_epoch, 10)

        network.train(data, target, epochs=5)
        self.assertEqual(network.last_epoch, 15)

    def test_train_and_test_dataset_training(self):
        data, target = datasets.make_classification(30, n_features=10,
                                                    n_classes=2)
        network = algorithms.GradientDescent((10, 3, 1))

        # Should work fine without exceptions
        network.train(data, target, epochs=2)
        network.train(data, target, data, target, epochs=2)

        with self.assertRaises(ValueError):
            network.train(data, target, data, epochs=2)

        with self.assertRaises(ValueError):
            network.train(data, target, target_test=target, epochs=2)

    def test_stop_iteration(self):
        def stop_training_after_the_5th_epoch(network):
            if network.last_epoch == 5:
                raise StopTraining("Stopped training")

        data, target = datasets.make_classification(30, n_features=10,
                                                    n_classes=2)
        network = algorithms.GradientDescent(
            (10, 3, 1),
            epoch_end_signal=stop_training_after_the_5th_epoch,
        )
        network.train(data, target, epochs=10)

        self.assertEqual(network.last_epoch, 5)

    def test_show_network_options_function(self):
        with catch_stdout() as out:
            # Disable verbose and than enable it again just
            # to make sure that `show_network_options` won't
            # trigger in the __init__ method
            network = algorithms.GradientDescent((2, 3, 1), verbose=False)
            network.verbose = True

            show_network_options(network)
            terminal_output = out.getvalue()

        self.assertIn('step', terminal_output)

    def test_logging_info_about_the_data(self):
        network = algorithms.GradientDescent((2, 3, 1))

        x = np.zeros((5, 2))
        x_test = np.zeros((3, 2))
        y = np.zeros((4, 1))

        with self.assertRaisesRegexp(ValueError, "feature shape"):
            logging_info_about_the_data(network, x, y)

        with catch_stdout() as out:
            network = algorithms.GradientDescent((2, 3, 1), verbose=True)
            logging_info_about_the_data(network, [x, x], [x_test, x_test])
            terminal_output = out.getvalue()

        self.assertIn("[(5, 2), (5, 2)]", terminal_output)
        self.assertIn("[(3, 2), (3, 2)]", terminal_output)

    def test_parse_show_epoch_property(self):
        with catch_stdout() as out:
            network = algorithms.GradientDescent(
                (2, 3, 1), show_epoch='5 times', verbose=True
            )

            show_epoch = parse_show_epoch_property(network, 100, epsilon=1e-2)
            self.assertEqual(show_epoch, 1)

            terminal_output = out.getvalue()

        self.assertIn("Can't use", terminal_output)

    def test_empty_error_history_list(self):
        errlist = ErrorHistoryList()
        norm_errlist = errlist.normalized()
        self.assertIs(norm_errlist, errlist)

    def test_network_train_epsilon_exception(self):
        network = algorithms.GradientDescent((2, 3, 1))

        x = np.zeros((5, 2))
        y = np.zeros((5, 1))

        with self.assertRaises(ValueError):
            network.train(x, y, epochs=-1)

        with self.assertRaises(ValueError):
            network.train(x, y, epsilon=1e-2, epochs=1)

    def test_network_training_with_unknown_summary_type(self):
        network = algorithms.GradientDescent((2, 3, 1))

        x = np.zeros((5, 2))
        y = np.zeros((5, 1))

        with self.assertRaises(ValueError):
            network.train(x, y, summary='unknown')

    def test_network_training_summary_inline(self):
        with catch_stdout() as out:
            network = algorithms.GradientDescent((2, 3, 1), verbose=False)

            x = np.zeros((5, 2))
            y = np.zeros((5, 1))

            network.verbose = True
            n_epochs = 10
            network.train(x, y, summary='inline', epochs=n_epochs)

            terminal_output = out.getvalue().strip()

        # `n_epochs - 1` because \n appears only between
        # inline summary lines.
        # Also network prints 5 additional lines at the beggining
        self.assertEqual(terminal_output.count('\n'), 5 + n_epochs - 1)


class NetworkPropertiesTestCase(BaseTestCase):
    def test_show_epoch_valid_cases(self):
        Case = namedtuple("Case", "show_epoch should_be_n_times n_epochs")
        cases = (
            # Show 10 epochs and the last one would be 11
            Case(show_epoch='10 times', should_be_n_times=11, n_epochs=100),
            Case(show_epoch='1 time', should_be_n_times=2, n_epochs=10),
            Case(show_epoch='1 times', should_be_n_times=2, n_epochs=10),
            # Should be equal to the number of epochs
            Case(show_epoch='100 times', should_be_n_times=10, n_epochs=10),
            Case(show_epoch=5, should_be_n_times=3, n_epochs=10),
            Case(show_epoch=100, should_be_n_times=2, n_epochs=10),
        )

        for case in cases:
            with catch_stdout() as out:
                bpnet = algorithms.GradientDescent(
                    (2, 3, 1),
                    step=0.1,
                    verbose=True,
                    show_epoch=case.show_epoch
                )
                bpnet.train(xor_zero_input_train, xor_zero_target_train,
                            epochs=case.n_epochs)
                terminal_output = out.getvalue()

            self.assertEqual(case.should_be_n_times,
                             terminal_output.count(" ms "))

    def test_show_epoch_invalid_cases(self):
        wrong_input_values = (
            'time 10', 'good time', '100', 'super power',
            '0 times', '-1 times',
            0, -100,
        )

        for wrong_input_value in wrong_input_values:
            with self.assertRaises(ValueError):
                algorithms.GradientDescent(
                    (2, 3, 1),
                    step=0.1,
                    verbose=False,
                    show_epoch=wrong_input_value
                )

    def test_network_convergence(self):
        with catch_stdout() as out:
            bpnet = algorithms.GradientDescent(
                (2, 3, 1),
                step=0.1,
                verbose=True,
                show_epoch=100
            )
            bpnet.train(xor_zero_input_train, xor_zero_target_train,
                        epochs=3, epsilon=1e-5)
            terminal_output = out.getvalue()
        self.assertEqual(1, terminal_output.count("Network didn't converge"))

        with catch_stdout() as out:
            bpnet = algorithms.GradientDescent(
                (2, 3, 1),
                step=0.1,
                verbose=True,
                show_epoch=100
            )
            bpnet.train(xor_zero_input_train, xor_zero_target_train,
                        epochs=1e3, epsilon=1e-3)
            terminal_output = out.getvalue()

        self.assertEqual(1, terminal_output.count("Network converged"))

    def test_terminal_output_frequency(self):
        with catch_stdout() as out:
            data = np.random.random((1000, 2))
            target = np.random.random((1000, 1))
            bpnet = algorithms.GradientDescent(
                (2, 1, 1),
                verbose=True,
                show_epoch=1
            )
            bpnet.train(data, target, epochs=100)
            terminal_output = out.getvalue()

        self.assertEqual(1, terminal_output.count("Too many outputs"))

    def test_network_architecture_output(self):
        expected_architecture = textwrap.dedent("""
        -----------------------------------------------
        | # | Input shape | Layer Type | Output shape |
        -----------------------------------------------
        | 1 | 2           | Input      | 2            |
        | 2 | 2           | Sigmoid    | 3            |
        | 3 | 3           | Sigmoid    | 1            |
        -----------------------------------------------
        """).strip()

        with catch_stdout() as out:
            network = algorithms.GradientDescent((2, 3, 1), verbose=True)
            network.architecture()

            terminal_output = out.getvalue().replace('\r', '')
            # Use assertTrue to make sure that it won't through
            # all variables in terminal in case of error
            self.assertTrue(expected_architecture in terminal_output)

    def test_network_architecture_output_exception(self):
        input_layer = layers.Input(10)
        hidden_layer_1 = layers.Sigmoid(20)
        hidden_layer_2 = layers.Sigmoid(20)
        output_layer = layers.Concatenate()

        connection = layers.join(input_layer, hidden_layer_1, output_layer)
        connection = layers.join(input_layer, hidden_layer_2, output_layer)

        network = algorithms.GradientDescent(connection)
        with self.assertRaises(TypeError):
            network.architecture()


class NetworkRepresentationTestCase(BaseTestCase):
    def test_small_network_representation(self):
        network = algorithms.GradientDescent((2, 3, 1))
        self.assertIn("Input(2) > Sigmoid(3) > Sigmoid(1)", str(network))

    def test_big_network_representation(self):
        network = algorithms.GradientDescent((2, 3, 4, 5, 6, 1))
        self.assertIn("[... 6 layers ...]", str(network))

    def test_network_representation_for_non_feedforward(self):
        input_layer = layers.Input(10)
        hidden_layer_1 = layers.Sigmoid(20)
        hidden_layer_2 = layers.Sigmoid(20)
        output_layer = layers.Concatenate()

        connection = layers.join(input_layer, hidden_layer_1, output_layer)
        connection = layers.join(input_layer, hidden_layer_2, output_layer)

        network = algorithms.GradientDescent(connection)
        self.assertIn("[... 4 layers ...]", str(network))
