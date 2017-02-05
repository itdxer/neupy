from functools import partial
from collections import namedtuple

import numpy as np
import theano
import theano.tensor as T

from neupy import algorithms
from neupy.utils import asfloat
import neupy.algorithms.gd.conjgrad as cg

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
                    np.array([0, 0]),
                ),
                answer=0.137
            ),
            Case(
                func=cg.polak_ribiere,
                input_data=(
                    np.array([1.,  -0.5]),
                    np.array([1.2, -0.45]),
                    np.array([0, 0]),
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
            input_data = asfloat(np.array(testcase.input_data))
            variables = T.vectors(3)
            # For functions some input variables can be optional and we
            # ignore them during the computation. This solution cause errors
            # related to the Theano computational graph, because we
            # do not use all defined variables. That's why we need
            # simple hack that fix this issue and do not add changes to
            # the output result.
            hack = asfloat(0) * variables[-1][0]
            output_func = theano.function(
                variables,
                testcase.func(*variables) + hack
            )
            result = output_func(*input_data)
            self.assertAlmostEqual(result, testcase.answer, places=1)

    def test_conjgrad(self):
        nw = algorithms.ConjugateGradient(
            self.connection,
            step=1,
            error='mse',
            shuffle_data=True,
            verbose=False,
            update_function='fletcher_reeves'
        )
        nw.train(simple_input_train, simple_target_train, epochs=100)
        result = nw.predict(simple_input_train)
        norm = np.linalg.norm(result - simple_target_train)
        self.assertAlmostEqual(0.05, norm, places=2)

    def test_compare_bp_and_cg(self):
        compare_networks(
            # Test classes
            algorithms.GradientDescent,
            partial(
                algorithms.ConjugateGradient,
                update_function='fletcher_reeves'
            ),
            # Test data
            (simple_input_train, simple_target_train),
            # Network configurations
            connection=self.connection,
            step=1,
            error='categorical_crossentropy',
            shuffle_data=True,
            # Test configurations
            epochs=50,
            show_comparison_plot=False
        )
