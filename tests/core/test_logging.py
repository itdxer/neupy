# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from collections import namedtuple

from neupy.core import logs
from neupy.core.logs import Verbose, TerminalLogger

from base import BaseTestCase
from helpers import catch_stdout


class LoggingTestCase(BaseTestCase):
    def test_logging_exceptions(self):
        with self.assertRaises(ValueError):
            logs = TerminalLogger()
            logs.message("tag", "text", color="unknown-color")

    def test_logging_switcher(self):
        class A(Verbose):
            def callme(self):
                self.logs.message("TEST", "output")

        with catch_stdout() as out:
            a = A(verbose=True)
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
        with catch_stdout() as out:
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
            )

            for test_case in test_cases:
                test_case.method(*test_case.msg_args)
                terminal_output = out.getvalue()
                self.assertRegexpMatches(
                    terminal_output, test_case.expectation)

    def test_terminal_colors(self):
        logger = TerminalLogger()
        real_is_color_supported = logs.is_color_supported

        logs.is_color_supported = lambda: False
        red_color = logger.colors['red']
        self.assertEqual('test', red_color('test'))

        logs.is_color_supported = lambda: True
        self.assertNotEqual('test', red_color('test'))
        self.assertIn('test', red_color('test'))

        logs.is_color_supported = real_is_color_supported
