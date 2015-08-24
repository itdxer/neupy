import logging
import random
import unittest

import numpy as np


DEFAULT_SEED = 0


class BaseTestCase(unittest.TestCase):
    verbose = False

    def setUp(self):
        np.random.seed(DEFAULT_SEED)
        random.seed(DEFAULT_SEED)

        if not self.verbose:
            logging.disable(logging.CRITICAL)

    def assertEqualArrays(self, actual, expected):
        self.assertTrue(np.all(actual == expected))
