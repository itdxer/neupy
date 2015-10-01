import random
import inspect
import logging
import unittest

import numpy as np
import pandas as pd


def create_vectors(vector, row1d=False):
    shape2d = (1, vector.size) if row1d else (vector.size, 1)

    vectors_list = []
    if vector.ndim == 1:
        vectors_list.extend([
            vector,
            pd.Series(vector)
        ])

    vectors_list.extend([
        vector.reshape(shape2d),
        pd.DataFrame(vector.reshape(shape2d))
    ])

    return vectors_list


class BaseTestCase(unittest.TestCase):
    verbose = False
    random_seed = 0

    def setUp(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        if not self.verbose:
            logging.disable(logging.CRITICAL)

    def assertInvalidVectorTrain(self, net, input_vector, target=None,
                                 decimal=5, row1d=False, **train_kwargs):

        test_vectors = create_vectors(input_vector, row1d=row1d)

        if target is not None:
            target_vectors = create_vectors(input_vector, row1d=row1d)
            test_vectors = zip(test_vectors, target_vectors)

        train_args = inspect.getargspec(net.train).args

        if 'epochs' in train_args and 'epochs' not in train_kwargs:
            train_kwargs['epochs'] = 5

        elif 'epsilon' in train_args and 'epsilon' not in train_kwargs:
            train_kwargs['epsilon'] = 0.1

        for i, test_args in enumerate(test_vectors, start=1):
            if target is None:
                net.train(test_args, **train_kwargs)
            else:
                net.train(*test_args, **train_kwargs)

    def assertInvalidVectorPred(self, net, input_vector, target, decimal=5,
                                row1d=False):
        test_vectors = create_vectors(input_vector, row1d=row1d)

        for i, test_vector in enumerate(test_vectors, start=1):
            np.testing.assert_array_almost_equal(
                net.predict(test_vector),
                target,
                decimal=decimal
            )
