from neupy import algorithms
from neupy.helpers import table
from neupy.algorithms.summary_info import InlineSummary, SummaryTable

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


class SummaryTableTestCase(BaseTestCase):
    def test_summary_table_delay_limit(self):
        with catch_stdout() as out:
            network = algorithms.GradientDescent((3, 2), verbose=True)
            network.train(simple_input_train, simple_target_train, epochs=20)

            terminal_output = out.getvalue()
            self.assertIn("Too many outputs", terminal_output)

    def test_summary_table_slow_training(self):
        with catch_stdout() as out:
            network = algorithms.GradientDescent((2, 3, 1), verbose=True)
            summary = SummaryTable(
                network,
                table_builder=table.TableBuilder(
                    table.Column(name="Epoch #"),
                    table.NumberColumn(name="Train err", places=4),
                    table.NumberColumn(name="Valid err", places=4),
                    table.TimeColumn(name="Time", width=10),
                    stdout=network.logs.write
                ),
                delay_limit=0,
                delay_history_length=1
            )

            for _ in range(3):
                network.training.epoch_time = 0.1
                summary.show_last()

            terminal_output = out.getvalue()
            self.assertNotIn("Too many outputs", terminal_output)
