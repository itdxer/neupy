import random

import theano
import numpy as np

from neupy import environment

from base import BaseTestCase


class EnvironmentTestCase(BaseTestCase):
    def test_speedup_environment(self):
        environment.speedup()

        self.assertEqual(theano.config.floatX, 'float32')
        self.assertEqual(theano.config.allow_gc, False)

    def test_reproducible_environment_math_library(self):
        environment.reproducible(seed=0)
        x1 = random.random()

        environment.reproducible(seed=0)
        x2 = random.random()

        self.assertAlmostEqual(x1, x2)

    def test_reproducible_environment_numpy_library(self):
        environment.reproducible(seed=0)
        x1 = np.random.random((10, 10))

        environment.reproducible(seed=0)
        x2 = np.random.random((10, 10))

        np.testing.assert_array_almost_equal(x1, x2)

    def test_sandbox_environment(self):
        environment.sandbox()

        self.assertEqual(theano.config.optimizer, 'fast_compile')
        self.assertEqual(theano.config.allow_gc, False)
