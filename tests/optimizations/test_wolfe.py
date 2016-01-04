from collections import namedtuple

import numpy as np

from neupy.optimizations import wolfe

from base import BaseTestCase


Case = namedtuple('Case', 'func_input func_expected')


class WolfeInterpolationTestCase(BaseTestCase):
    def assertAlmostEqual(self, left_value, right_value, places=6):
        if np.isnan(left_value) or np.isnan(right_value):
            self.assertTrue(np.isnan(left_value) and np.isnan(right_value))
        else:
            super(WolfeInterpolationTestCase, self).assertAlmostEqual(
                left_value, right_value, places
            )

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
            actual_output = wolfe.quadratic_minimizer(**testcase.func_input)
            self.assertAlmostEqual(actual_output.eval(),
                                   testcase.func_expected)

    def test_cubic_minimizer(self):
        testcases = (
            Case(func_input=dict(x_a=0, y_a=1, y_prime_a=-1,
                                 x_b=1, y_b=2, x_c=2, y_c=3),
                 func_expected=0.18),
        )

        for testcase in testcases:
            actual_output = wolfe.cubic_minimizer(**testcase.func_input)
            self.assertAlmostEqual(actual_output.eval(),
                                   testcase.func_expected,
                                   places=2)
