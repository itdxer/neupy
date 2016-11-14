import time

import six
import numpy as np

from neupy.helpers import Progressbar
from neupy.helpers.progressbar import format_time, FormatInlineDict
from neupy.algorithms.gd.base import format_error

from base import BaseTestCase
from utils import catch_stdout


class ProgressbarTestCase(BaseTestCase):
    def test_progressbar(self):
        out = six.StringIO()
        iterator = Progressbar(range(10), file=out)

        for i in iterator:
            time.sleep(0.1)
            terminal_output = out.getvalue()

            self.assertRegexpMatches(
                terminal_output,
                '\|{}{}\|\s{}/10\s+{}\%.+'.format(
                    '#' * i, '-' * (10 - i), i, i * 10
                )
            )

    def test_progressbar_with_long_update_freq(self):
        with catch_stdout() as out:
            iterator = Progressbar(range(10), file=out, update_freq=100)

            for i in iterator:
                pass

            terminal_output = out.getvalue()

        self.assertIn('0/10', terminal_output)

    def test_time_format(self):
        testcases = (
            dict(time_in_seconds=5, expected_format='00:05'),
            dict(time_in_seconds=62, expected_format='01:02'),
            dict(time_in_seconds=59 * 61, expected_format='59:59'),
            dict(time_in_seconds=60 * 60, expected_format='01:00:00'),
            dict(time_in_seconds=30 * 60 * 60, expected_format='30:00:00'),
        )

        for testcase in testcases:
            actual_format = format_time(testcase['time_in_seconds'])
            self.assertEqual(actual_format, testcase['expected_format'])

    def test_format_error(self):
        testcases = (
            dict(error=None, expected_format='?'),
            dict(error=1., expected_format='1.00000'),
            dict(error=1 / 3., expected_format='0.33333'),
            dict(error=np.array([0.456789]), expected_format='0.45679'),
        )

        for testcase in testcases:
            actual_format = format_error(testcase['error'])
            self.assertEqual(actual_format, testcase['expected_format'])

    def test_inline_format_dict(self):
        data = FormatInlineDict([('bca', 1), ('abc', 2), (None, 3)])
        formated_data = '{}'.format(data)
        self.assertEqual(formated_data, 'bca: 1 abc: 2 None: 3')
