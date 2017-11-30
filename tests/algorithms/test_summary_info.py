from neupy import algorithms
from neupy.algorithms.summary_info import (InlineSummary, SummaryTable,
                                           format_time)

from base import BaseTestCase
from utils import catch_stdout
from data import simple_input_train, simple_target_train


class InlineSummaryTestCase(BaseTestCase):
    def test_inline_summary_with_validation(self):
        with catch_stdout() as out:
            network = algorithms.GradientDescent((2, 3, 1), verbose=True)
            summary = InlineSummary(network)

            network.last_epoch = 12
            network.training.epoch_time = 0.1
            network.errors.append(10)
            network.validation_errors.append(20)

            summary.show_last()
            summary.finish()

            terminal_output = out.getvalue()

            # training error is 10
            self.assertIn("10", terminal_output)
            # validation error is 20
            self.assertIn("20", terminal_output)
            # 0.1 sec
            self.assertIn("0.1", terminal_output)
            # 12th epoch
            self.assertIn("12", terminal_output)

    def test_inline_summary_without_validation(self):
        with catch_stdout() as out:
            network = algorithms.GradientDescent((2, 3, 1), verbose=True)
            summary = InlineSummary(network)

            network.last_epoch = 12
            network.training.epoch_time = 0.1
            network.errors.append(10)
            network.validation_errors.append(None)

            summary.show_last()
            terminal_output = out.getvalue()

            # training error is 10
            self.assertIn("10", terminal_output)
            # 0.1 sec
            self.assertIn("0.1", terminal_output)
            # 12th epoch
            self.assertIn("12", terminal_output)
            # No reference to validation error in the last line
            output_lines = terminal_output.split('\n')
            last_output_line = output_lines[-2]
            self.assertNotIn("None", last_output_line)

    def test_format_time(self):
        self.assertEqual("01:06:40", format_time(4000))
        self.assertEqual("02:05", format_time(125))
        self.assertEqual("45 sec", format_time(45))
        self.assertEqual("100 ms", format_time(0.1))
        self.assertEqual("10 \u03BCs", format_time(1e-5))
        self.assertEqual("200 ns", format_time(2e-7))
