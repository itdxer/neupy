import random
import inspect
import logging
import unittest

import numpy as np
import pandas as pd


def create_vectors(vector, rows1d=False):
    shape2d = (1, len(vector)) if rows1d else (len(vector), 1)
    return [
        vector,
        vector.reshape(shape2d),
        pd.DataFrame(vector.reshape(shape2d)),
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

    def assertInvalidVectorTrain(self, net, input_vector, target=None,
                                 decimal=5, rows1d=False, **train_kwargs):
        test_vectors = create_vectors(input_vector, rows1d=rows1d)

        if target is not None:
            target_vectors = create_vectors(input_vector, rows1d=rows1d)
            test_vectors = zip(test_vectors, target_vectors)

        train_args = inspect.getargspec(net.train).args
        if 'epochs' in train_args and 'epochs' not in train_kwargs:
            train_kwargs['epochs'] = 5

        for test_args in test_vectors:
            if target is None:
                net.train(test_args, **train_kwargs)
            else:
                net.train(*test_args, **train_kwargs)

    def assertInvalidVectorPred(self, net, input_vector, target, decimal=5,
                                rows1d=False):
        test_vectors = create_vectors(input_vector, rows1d=rows1d)

        for test_vector in test_vectors:
            np.testing.assert_array_almost_equal(
                net.predict(test_vector),
                target,
                decimal=decimal
            )
