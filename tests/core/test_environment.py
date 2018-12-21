import random

import numpy as np

from neupy import utils

from base import BaseTestCase


class EnvironmentTestCase(BaseTestCase):

    def test_reproducible_utils_math_library(self):
        utils.reproducible(seed=0)
        x1 = random.random()

        utils.reproducible(seed=0)
        x2 = random.random()

        self.assertAlmostEqual(x1, x2)

    def test_reproducible_utils_numpy_library(self):
        utils.reproducible(seed=0)
        x1 = np.random.random((10, 10))

        utils.reproducible(seed=0)
        x2 = np.random.random((10, 10))

        np.testing.assert_array_almost_equal(x1, x2)
