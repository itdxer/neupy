from collections import namedtuple

import numpy as np

from neupy.optimizations import wolfe

from base import BaseTestCase


Case = namedtuple('Case', 'func_input func_expected')


class WolfeInterpolationTestCase(BaseTestCase):
    def test_quadratic_minimizer(self):
        nan = np.array(np.nan)
        testcases = (
            Case(func_input=dict(x_a=0, y_a=1, y_prime_a=-1, x_b=1, y_b=2),
                 func_expected=0.25),
            Case(func_input=dict(x_a=1, y_a=1, y_prime_a=-1, x_b=2, y_b=2),
                 func_expected=1.25),
            # For x_b < x_a
            Case(func_input=dict(x_a=1, y_a=1, y_prime_a=-1, x_b=0, y_b=2),
                 func_expected=nan),
            # Slope at point ``x`` shows that function is upside down
            Case(func_input=dict(x_a=0, y_a=1, y_prime_a=1, x_b=1, y_b=2),
                 func_expected=nan),
        )

        for testcase in testcases:
            actual_output = wolfe.quadratic_minimizer(
                **testcase.func_input
            ).eval()

            if np.isnan(testcase.func_expected):
                self.assertTrue(
                    np.isnan(testcase.func_expected) and
                    np.isnan(actual_output)
                )
            else:
                self.assertEqual(actual_output, testcase.func_expected)
