import logging
import random
import unittest

import numpy as np


class BaseTestCase(unittest.TestCase):
    verbose = False
    random_seed = 0

    def setUp(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        if not self.verbose:
            logging.disable(logging.CRITICAL)

    def assertEqualArrays(self, actual, expected):
        self.assertTrue(np.all(actual == expected))
