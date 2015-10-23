import sys
from contextlib import contextmanager
from collections import namedtuple

import six
from neupy import algorithms

from data import xor_zero_input_train, xor_zero_target_train
from base import BaseTestCase


@contextmanager
def catch_stdout():
    old_out = sys.stdout
    out = six.StringIO()
    sys.stdout = out

    yield out

    sys.stdout = old_out


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
                bpnet = algorithms.Backpropagation(
                    (2, 3, 1),
                    step=0.1,
                    verbose=True,
                    show_epoch=case.show_epoch
                )
                bpnet.train(xor_zero_input_train, xor_zero_target_train,
                            epochs=case.n_epochs)
                terminal_output = out.getvalue()

            self.assertEqual(case.should_be_n_times,
                             terminal_output.count("Train error"))

    def test_show_epoch_invalid_cases(self):
        wrong_input_values = (
            'time 10', 'good time', '100', 'super power',
            '0 times', '-1 times',
            0, -100,
        )

        for wrong_input_value in wrong_input_values:
            with self.assertRaises(ValueError):
                bpnet = algorithms.Backpropagation(
                    (2, 3, 1),
                    step=0.1,
                    verbose=False,
                    show_epoch=wrong_input_value
                )
