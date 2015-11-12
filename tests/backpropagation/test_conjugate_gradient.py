from functools import partial
from collections import namedtuple

import numpy as np

from neupy import algorithms
import neupy.algorithms.backprop.conjugate_gradient as cg
from neupy.functions import cross_entropy_error
from neupy.layers import *

from data import simple_input_train, simple_target_train
from utils import compare_networks
from base import BaseTestCase


class ConjugateGradientTestCase(BaseTestCase):
    def setUp(self):
        super(ConjugateGradientTestCase, self).setUp()
        self.connection = (3, 5, 2)

    def test_functions(self):
        Case = namedtuple("Case", "func input_data answer")

        testcases = [
            Case(
                func=cg.fletcher_reeves,
                input_data=(
                    np.array([1.35,  0.3]),
                    np.array([0.11, -0.5]),
                    None
                ),
                answer=0.137
            ),
            Case(
                func=cg.polak_ribiere,
                input_data=(
                    np.array([1.,  -0.5]),
                    np.array([1.2, -0.45]),
                    None
                ),
                answer=0.174
            ),
            Case(
                func=cg.hentenes_stiefel,
                input_data=(
                    np.array([1.,  -0.5]),
                    np.array([1.2, -0.45]),
                    np.array([0.2, 0.05]),
                ),
                answer=5.118
            ),
            Case(
                func=cg.conjugate_descent,
                input_data=(
                    np.array([1.,  -0.5]),
                    np.array([1.2, -0.45]),
                    np.array([0.2, 0.05]),
                ),
                answer=-7.323
            ),
            Case(
                func=cg.liu_storey,
                input_data=(
                    np.array([1.,  -0.5]),
                    np.array([1.2, -0.45]),
                    np.array([0.2, 0.05]),
                ),
                answer=1.243
            ),
            Case(
                func=cg.dai_yuan,
                input_data=(
                    np.array([1.,  -0.5]),
                    np.array([1.2, -0.45]),
                    np.array([0.2, 0.05]),
                ),
                answer=38.647
            ),
        ]

        for testcase in testcases:
            result = testcase.func(*testcase.input_data)
            self.assertAlmostEqual(result, testcase.answer, places=3)


    def test_conjugate_gradient(self):
        nw = algorithms.ConjugateGradient(
            self.connection,
            step=5,
            error=cross_entropy_error,
            shuffle_data=True,
            update_function='polak_ribiere'
        )
        nw.train(simple_input_train, simple_target_train, epochs=300)
        result = nw.predict(simple_input_train)
        norm = np.linalg.norm(result - simple_target_train)
        self.assertGreater(1e-2, norm)

    def test_compare_bp_and_cg(self):
        compare_networks(
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
            shuffle_data=True,
            # Test configurations
            epochs=50,
            # is_comparison_plot=True
        )
