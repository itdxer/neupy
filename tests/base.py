import inspect
import logging
import unittest

import numpy as np

from neupy import environment, layers

from utils import vectors_for_testing


class BaseTestCase(unittest.TestCase):
    verbose = False
    random_seed = 0
    use_sandbox_mode = True

    def setUp(self):
        environment.reproducible(seed=self.random_seed)

        if not self.verbose:
            logging.disable(logging.CRITICAL)

        if self.use_sandbox_mode:
            # Optimize unit tests speed. In general all task very
            # simple so some Theano optimizations can be redundant.
            environment.sandbox()

        # Clean identifiers map for each test
        layers.BaseLayer.global_identifiers_map = {}

    def assertInvalidVectorTrain(self, network, input_vector, target=None,
                                 decimal=5, is_feature1d=True, **train_kwargs):
        """
        Method helps test network prediction training using different
        types of row or column vector.
        """
        input_vectors = vectors_for_testing(input_vector, is_feature1d)

        if target is not None:
            target_vectors = vectors_for_testing(target, is_feature1d)
            input_vectors = zip(input_vectors, target_vectors)

        train_args = inspect.getargspec(network.train).args

        if 'epochs' in train_args and 'epochs' not in train_kwargs:
            train_kwargs['epochs'] = 5

        elif 'epsilon' in train_args and 'epsilon' not in train_kwargs:
            train_kwargs['epsilon'] = 0.1

        for i, input_data in enumerate(input_vectors, start=1):
            if target is None:
                network.train(input_data, **train_kwargs)
            else:
                network.train(*input_data, **train_kwargs)

    def assertInvalidVectorPred(self, network, input_vector, target,
                                decimal=5, is_feature1d=True):
        """
        Method helps test network prediction procedure using different
        types of row or column vector.
        """
        test_vectors = vectors_for_testing(input_vector, is_feature1d)

        for i, test_vector in enumerate(test_vectors, start=1):
            predicted_vector = network.predict(test_vector)
            np.testing.assert_array_almost_equal(predicted_vector, target,
                                                 decimal=decimal)
