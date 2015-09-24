import logging
import random
import unittest

import numpy as np
import pandas as pd


def create_test_vectors(vector):
    return [
        vector,
        vector.reshape((len(vector), 1)),
        pd.DataFrame(vector),
        pd.Series(vector)
    ]

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

    def assertInvalidVectorTrain(self, net, input_vector, target=None,
                                 decimal=5, **train_kwargs):
        test_vectors = create_test_vectors(input_vector)
        if target is not None:
            test_vectors = zip(test_vectors,
                               create_test_vectors(input_vector))

        if 'epochs' not in train_kwargs:
            train_kwargs['epochs'] = 5

        for test_args in test_vectors:
            if target is None:
                net.train(test_args, **train_kwargs)
            else:
                net.train(*test_args, **train_kwargs)

    def assertInvalidVectorPred(self, net, input_vector, target, decimal=5):
        test_vectors = create_test_vectors(input_vector)

        for test_vector in test_vectors:
            np.testing.assert_array_almost_equal(
                net.predict(test_vector),
                target,
                decimal=decimal
            )
