import pickle
import inspect
import logging
import unittest

import numpy as np
import tensorflow as tf

from neupy import environment, layers
from neupy.utils import (
    tensorflow_eval,
    tensorflow_session,
    initialize_uninitialized_variables,
)

from utils import vectors_for_testing


class BaseTestCase(unittest.TestCase):
    single_thread = False
    verbose = False
    random_seed = 0

    def eval(self, value):
        return tensorflow_eval(value)

    def setUp(self):
        tf.reset_default_graph()

        if self.single_thread:
            sess = tensorflow_session()
            sess.close()

            config = tf.ConfigProto(
                allow_soft_placement=True,
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1,
            )
            tensorflow_session.cache = tf.Session(config=config)

        if not self.verbose:
            logging.disable(logging.CRITICAL)

        # Clean identifiers map for each test
        layers.BaseLayer.global_identifiers_map = {}
        environment.reproducible(seed=self.random_seed)

    def tearDown(self):
        sess = tensorflow_session()
        sess.close()

    def assertItemsEqual(self, list1, list2):
        self.assertEqual(sorted(list1), sorted(list2))

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

    def assertPickledNetwork(self, network, input_data):
        stored_network = pickle.dumps(network)
        loaded_network = pickle.loads(stored_network)

        network_prediction = network.predict(input_data)
        loaded_network_prediction = loaded_network.predict(input_data)

        np.testing.assert_array_almost_equal(
            loaded_network_prediction, network_prediction)
