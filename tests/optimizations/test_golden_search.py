import theano.tensor as T

from neupy.optimizations.golden_search import fmin_golden_search

from base import BaseTestCase


class GoldenSearchTestCase(BaseTestCase):
    def test_golden_search_exceptions(self):
        invalid_parameters = (
            dict(tol=-1),
            dict(minstep=-1),
            dict(maxstep=-1),
            dict(maxiter=-1),
            dict(tol=0),
            dict(minstep=0),
            dict(maxstep=0),
            dict(maxiter=0),
        )
        for params in invalid_parameters:
            with self.assertRaises(ValueError):
                fmin_golden_search(lambda x: x, **params)

        with self.assertRaises(ValueError):
            fmin_golden_search(lambda x: x, minstep=10, maxstep=1)

    def test_golden_search_function(self):
        def f(x):
            return T.sin(x) * x ** -0.5

        def check_updates(step):
            return f(3 + step)

        best_step = fmin_golden_search(check_updates)
        self.assertAlmostEqual(1.6, best_step.eval(), places=2)

        best_step = fmin_golden_search(check_updates, maxstep=1)
        self.assertAlmostEqual(1, best_step.eval(), places=2)
