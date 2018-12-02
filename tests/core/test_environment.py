import random

import numpy as np

from neupy import environment

from base import BaseTestCase


class EnvironmentTestCase(BaseTestCase):

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
