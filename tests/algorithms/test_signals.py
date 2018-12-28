# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from collections import namedtuple

import numpy as np
from sklearn.datasets import make_classification

from neupy import algorithms, layers
from neupy.exceptions import StopTraining
from neupy.algorithms.signals import format_time

from base import BaseTestCase
from helpers import catch_stdout


def train_network(epochs=2, **kwargs):
    network = algorithms.GradientDescent(
        [
            layers.Input(10),
            layers.Sigmoid(3),
            layers.Sigmoid(1),
        ],
        **kwargs
    )

    data, target = make_classification(30, n_features=10, n_classes=2)
    network.train(data, target, data, target, epochs=epochs)

    return network


class SignalsTestCase(BaseTestCase):
    def test_signal_func_one_training_update_end(self):
        global triggered_times
        triggered_times = 0

        def print_message(network):
            global triggered_times
            triggered_times += 1

        train_network(
            epochs=4,
            signals=print_message,
            batch_size=None)

        self.assertEqual(triggered_times, 4)

    def test_signal_func_stop_iteration(self):
        def stop_training_after_the_5th_epoch(network):
            if network.last_epoch == 5:
                raise StopTraining("Stopped training")

        network = train_network(
            epochs=10,
            batch_size=None,
            signals=stop_training_after_the_5th_epoch)

        self.assertEqual(network.last_epoch, 5)

    def test_custom_signal_class(self):
        class SimpleSignal(object):
            triggers = []

            n_train_start = 0
            n_train_end = 0

            n_epoch_start = 0
            n_epoch_end = 0

            n_update_start = 0
            n_update_end = 0

            n_train_error = 0
            n_valid_error = 0

            def train_start(self, network, **data):
                self.n_train_start += 1
                self.triggers.append('train_start')

            def train_end(self, network):
                self.n_train_end += 1
                self.triggers.append('train_end')

            def epoch_start(self, network):
                self.n_epoch_start += 1
                self.triggers.append('epoch_start')

            def epoch_end(self, network):
                self.n_epoch_end += 1
                self.triggers.append('epoch_end')

            def update_start(self, network):
                self.n_update_start += 1
                self.triggers.append('update_start')

            def update_end(self, network):
                self.n_update_end += 1
                self.triggers.append('update_end')

            def train_error(self, network, **data):
                self.n_train_error += 1
                self.triggers.append('train_error')

            def valid_error(self, network, **data):
                self.n_valid_error += 1
                self.triggers.append('valid_error')

        simple_signal = SimpleSignal()
        train_network(signals=[simple_signal], batch_size=10)

        self.assertEqual(simple_signal.n_train_start, 1)
        self.assertEqual(simple_signal.n_train_end, 1)

        self.assertEqual(simple_signal.n_epoch_start, 2)
        self.assertEqual(simple_signal.n_epoch_end, 2)

        self.assertEqual(simple_signal.n_update_start, 6)
        self.assertEqual(simple_signal.n_update_end, 6)

        expected_triggers = [
            'train_start',

            'epoch_start',
            'update_start', 'train_error', 'update_end',
            'update_start', 'train_error', 'update_end',
            'update_start', 'train_error', 'update_end',
            'valid_error',
            'epoch_end',

            'epoch_start',
            'update_start', 'train_error', 'update_end',
            'update_start', 'train_error', 'update_end',
            'update_start', 'train_error', 'update_end',
            'valid_error',
            'epoch_end',

            'train_end',
        ]
        self.assertSequenceEqual(expected_triggers, simple_signal.triggers)

    def test_multiple_signals(self):
        class SimpleSignal(object):
            events = []

            def __init__(self, name):
                self.name = name

            def update_end(self, network):
                self.events.append(self.name)

        train_network(
            batch_size=10,
            signals=[SimpleSignal('a'), SimpleSignal('b')])

        self.assertEqual(SimpleSignal.events, ['a', 'b'] * 6)

    def test_print_training_progress_signal(self):
        x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_train = np.array([[1, 0, 0, 1]]).T

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
                bpnet.train(x_train, y_train, epochs=case.n_epochs)
                terminal_output = out.getvalue()

            # One of the choices has to be true whether other
            # choices should give count equal to zero.
            time_counts = (
                terminal_output.count(" μs]") +
                terminal_output.count(" ms]") +
                terminal_output.count(" ns]")
            )
            self.assertEqual(case.should_be_n_times, time_counts)

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
        self.assertEqual(terminal_output.count('train: '), n_epochs)

    def test_format_time(self):
        self.assertEqual("01:06:40", format_time(4000))
        self.assertEqual("02:05", format_time(125))
        self.assertEqual("45 sec", format_time(45))
        self.assertEqual("100 ms", format_time(0.1))
        self.assertEqual("10 μs", format_time(1e-5))
        self.assertEqual("200 ns", format_time(2e-7))
