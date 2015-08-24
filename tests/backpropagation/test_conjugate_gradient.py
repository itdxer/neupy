from functools import partial

import numpy as np

from neuralpy import algorithms
import neuralpy.algorithms.backprop.conjugate_gradient as cg
from neuralpy.functions import cross_entropy_error
from neuralpy.layers import *

from data import simple_input_train, simple_target_train
from utils import compare_networks
from base import BaseTestCase


class ConjugateGradientTestCase(BaseTestCase):
    def setUp(self):
        super(ConjugateGradientTestCase, self).setUp()
        self.connection = (3, 5, 2)

    def test_functions(self):
        g_new = np.matrix([[0.11, -0.5]]).T
        g_old = np.matrix([[1.35,  0.3]]).T
        result = cg.fletcher_reeves(g_old, g_new, None)
        self.assertEqual(np.round(result, 3), 0.137)

    def test_conjugate_gradient(self):
        nw = algorithms.ConjugateGradient(
            self.connection,
            step=5,
            error=cross_entropy_error,
            use_raw_predict_at_error=False,
            shuffle_data=True,
            update_function='polak_ribiere'
        )
        nw.train(simple_input_train, simple_target_train, epochs=300)
        result = nw.predict(simple_input_train)
        norm = np.linalg.norm(result - simple_target_train)
        self.assertGreater(1e-2, norm)

    def test_compare_bp_and_cg(self):
        network_default_error, network_tested_error = compare_networks(
            # Test classes
            algorithms.Backpropagation,
            partial(
                algorithms.ConjugateGradient,
                update_function='fletcher_reeves'
            ),
            # Test data
            (simple_input_train, simple_target_train),
            # Network configurations
            connection=self.connection,
            step=1,
            error=cross_entropy_error,
            use_raw_predict_at_error=False,
            shuffle_data=True,
            # Test configurations
            epochs=50,
            # is_comparison_plot=True
        )
        self.assertGreater(network_default_error, network_tested_error)
