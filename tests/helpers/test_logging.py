from collections import namedtuple

from neupy.helpers.logs import Verbose, TerminalLogger
from neupy import algorithms

from base import BaseTestCase
from utils import catch_stdout
from data import simple_classification


class LoggingTestCase(BaseTestCase):
    def test_logging_switcher(self):
        class A(Verbose):
            def callme(self):
                self.logs.message("TEST", "output")

        a = A(verbose=True)
        with catch_stdout() as out:
            a.callme()
            terminal_output = out.getvalue()

            self.assertIn("TEST", terminal_output)
            self.assertIn("output", terminal_output)

        a.verbose = False
        with catch_stdout() as out:
            a.callme()
            terminal_output = out.getvalue()

            self.assertNotIn("TEST", terminal_output)
            self.assertNotIn("output", terminal_output)

    def test_logging_methods(self):
        logs = TerminalLogger()

        Case = namedtuple("Case", "method msg_args expectation")
        test_cases = (
            Case(
                logs.write,
                msg_args=["Simple text"],
                expectation="Simple text"
            ),
            Case(
                logs.message,
                msg_args=["TEST", "Message"],
                expectation=r"\[.*TEST.*\] Message",
            ),
            Case(
                logs.title,
                msg_args=["Title"],
                expectation=r"\n.*Title.*\n",
            ),
            Case(
                logs.error,
                msg_args=["Error message"],
                expectation=r"\[.*ERROR.*\] Error message",
            ),
            Case(
                logs.warning,
                msg_args=["Warning message"],
                expectation=r"\[.*WARN.*\] Warning message",
            )
        )

        for test_case in test_cases:
            with catch_stdout() as out:
                test_case.method(*test_case.msg_args)
                terminal_output = out.getvalue()
                self.assertRegexpMatches(terminal_output,
                                         test_case.expectation)


class NeuralNetworkLoggingTestCase(BaseTestCase):
    def test_nn_init_logging(self):
        with catch_stdout() as out:
            gdnet = algorithms.GradientDescent((2, 3, 1), verbose=False)
            terminal_output = out.getvalue()
            self.assertEqual("", terminal_output.strip())

        with catch_stdout() as out:
            gdnet = algorithms.GradientDescent((2, 3, 1), verbose=True)
            terminal_output = out.getvalue()

            self.assertNotEqual("", terminal_output.strip())
            self.assertIn("verbose = True", terminal_output)

    def test_nn_training(self):
        x_train, x_test, y_train, y_test = simple_classification()

        with catch_stdout() as out:
            gdnet = algorithms.GradientDescent((10, 20, 1), verbose=True)
            gdnet.train(x_train, y_train, x_test, y_test, epochs=4)

            terminal_output = out.getvalue()

            self.assertIn("Start train", terminal_output)
            self.assertIn("-----", terminal_output)
