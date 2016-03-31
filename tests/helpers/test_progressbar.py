import sys
import time

from neupy.helpers import progressbar

from base import BaseTestCase
from utils import catch_stdout


class ProgressbarTestCase(BaseTestCase):
    def test_simple_progressbar(self):
        with catch_stdout() as out:
            iterator = progressbar(
                range(10),
                mininterval=0.,
                file=sys.stdout,
                init_interval=0.
            )

            for i in iterator:
                time.sleep(0.1)
                terminal_output = out.getvalue()

                self.assertRegexpMatches(
                    terminal_output,
                    '\|{}{}\|\s{}/10\s+{}\%.+'.format(
                        '#' * i, '-' * (10 - i), i, i * 10
                    )
                )
