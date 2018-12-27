# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from collections import namedtuple

import numpy as np
from sklearn import datasets
from neupy import algorithms, layers
from neupy.exceptions import StopTraining
from neupy.algorithms.signals import format_time

from utils import catch_stdout
from base import BaseTestCase


xor_zero_x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_zero_y_train = np.array([[1, 0, 0, 1]]).T


class NetworkMainTestCase(BaseTestCase):
    def test_training_epoch_accumulation(self):
        data, target = datasets.make_classification(
            30, n_features=10, n_classes=2)

        network = algorithms.GradientDescent([
            layers.Input(10),
            layers.Sigmoid(3),
            layers.Sigmoid(1),
        ])
        self.assertEqual(network.last_epoch, 0)

        network.train(data, target, epochs=10)
        self.assertEqual(network.last_epoch, 10)

        network.train(data, target, epochs=5)
        self.assertEqual(network.last_epoch, 15)

    def test_train_and_test_dataset_training(self):
        data, target = datasets.make_classification(
            30, n_features=10, n_classes=2)

        network = algorithms.GradientDescent([
                layers.Input(10),
                layers.Sigmoid(3),
                layers.Sigmoid(1),
            ],
            batch_size=None,
        )

        # Should work fine without exceptions
        network.train(data, target, epochs=2)
        network.train(data, target, data, target, epochs=2)

        with self.assertRaises(ValueError):
            network.train(data, target, data, epochs=2)

        with self.assertRaises(ValueError):
            network.train(data, target, y_test=target, epochs=2)

    def test_stop_iteration(self):
        def stop_training_after_the_5th_epoch(network):
            if network.last_epoch == 5:
                raise StopTraining("Stopped training")

        data, target = datasets.make_classification(
            30, n_features=10, n_classes=2)

        network = algorithms.GradientDescent(
            [
                layers.Input(10),
                layers.Sigmoid(3),
                layers.Sigmoid(1),
            ],
            batch_size=None,
            signals=stop_training_after_the_5th_epoch,
        )
        network.train(data, target, epochs=10)
        self.assertEqual(network.last_epoch, 5)

    def test_network_training_summary(self):
        with catch_stdout() as out:
            network = algorithms.GradientDescent(
                layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1),
                verbose=False,
                batch_size=None,
            )

            x = np.zeros((5, 2))
            y = np.zeros((5, 1))

            network.verbose = True
            n_epochs = 10
            network.train(x, y, epochs=n_epochs)

            terminal_output = out.getvalue().strip()

        # `n_epochs - 1` because \n appears only between
        # inline summary lines.
        # Also network prints 5 additional lines at the beggining
        self.assertEqual(terminal_output.count('\n'), n_epochs - 1)

    def test_wrong_number_of_training_epochs(self):
        network = algorithms.GradientDescent(
            layers.Input(2) > layers.Sigmoid(1),
            verbose=False,
            batch_size=None,
        )

        with self.assertRaisesRegexp(ValueError, "a positive number"):
            network.train(np.zeros((4, 2)), np.zeros((4, 1)), epochs=0)

        with self.assertRaisesRegexp(ValueError, "a positive number"):
            network.train(np.zeros((4, 2)), np.zeros((4, 1)), epochs=-1)

    def test_format_time(self):
        self.assertEqual("01:06:40", format_time(4000))
        self.assertEqual("02:05", format_time(125))
        self.assertEqual("45 sec", format_time(45))
        self.assertEqual("100 ms", format_time(0.1))
        self.assertEqual("10 μs", format_time(1e-5))
        self.assertEqual("200 ns", format_time(2e-7))


class NetworkPropertiesTestCase(BaseTestCase):
    def test_show_epoch_valid_cases(self):
        Case = namedtuple("Case", "show_epoch should_be_n_times n_epochs")
        cases = (
            Case(show_epoch=5, should_be_n_times=3, n_epochs=10),
            Case(show_epoch=7, should_be_n_times=3, n_epochs=10),
            Case(show_epoch=100, should_be_n_times=2, n_epochs=10),
        )

        for case in cases:
            with catch_stdout() as out:
                bpnet = algorithms.GradientDescent(
                    layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1),
                    step=0.1,
                    verbose=True,
                    batch_size=None,
                    show_epoch=case.show_epoch
                )
                bpnet.train(
                    xor_zero_x_train,
                    xor_zero_y_train,
                    epochs=case.n_epochs,
                )
                terminal_output = out.getvalue()

            # One of the choices has to be true whether other
            # choices should give count equal to zero.
            time_counts = (
                terminal_output.count(" μs]") +
                terminal_output.count(" ms]") +
                terminal_output.count(" ns]")
            )
            self.assertEqual(case.should_be_n_times, time_counts)

    def test_show_epoch_invalid_cases(self):
        network = layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1)

        for wrong_value in (0, -100):
            with self.assertRaises(ValueError):
                algorithms.GradientDescent(network, show_epoch=wrong_value)

    def test_one_training_update_end(self):
        global triggered_times

        triggered_times = 0
        epochs = 4

        def print_message(network):
            global triggered_times
            triggered_times += 1

        network = algorithms.GradientDescent(
            connection=[
                layers.Input(2),
                layers.Sigmoid(2),
                layers.Sigmoid(1)
            ],
            signals=print_message,
            batch_size=None,
        )

        network.train(xor_zero_x_train, xor_zero_y_train, epochs=epochs)
        self.assertEqual(triggered_times, epochs)
