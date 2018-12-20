from functools import partial
from collections import namedtuple

import numpy as np
from sklearn import metrics

from neupy import algorithms, layers
from neupy.utils import asfloat
from neupy.algorithms.gd import conjgrad as cg

from data import simple_classification
from utils import compare_networks
from base import BaseTestCase


class ConjugateGradientTestCase(BaseTestCase):
    def test_functions(self):
        Case = namedtuple("Case", "func input_data answer")

        testcases = [
            Case(
                func=cg.fletcher_reeves,
                input_data=(
                    asfloat(np.array([1.35,  0.3])),
                    asfloat(np.array([0.11, -0.5])),
                    asfloat(np.array([0, 0])),
                ),
                answer=0.137
            ),
            Case(
                func=cg.polak_ribiere,
                input_data=(
                    asfloat(np.array([1.,  -0.5])),
                    asfloat(np.array([1.2, -0.45])),
                    asfloat(np.array([0, 0])),
                ),
                answer=0.174
            ),
            Case(
                func=cg.hentenes_stiefel,
                input_data=(
                    asfloat(np.array([1.,  -0.5])),
                    asfloat(np.array([1.2, -0.45])),
                    asfloat(np.array([0.2, 0.05])),
                ),
                answer=5.118
            ),
            Case(
                func=cg.liu_storey,
                input_data=(
                    asfloat(np.array([1.,  -0.5])),
                    asfloat(np.array([1.2, -0.45])),
                    asfloat(np.array([0.2, 0.05])),
                ),
                answer=-1.243
            ),
            Case(
                func=cg.dai_yuan,
                input_data=(
                    asfloat(np.array([1.,  -0.5])),
                    asfloat(np.array([1.2, -0.45])),
                    asfloat(np.array([0.2, 0.05])),
                ),
                answer=38.647
            ),
        ]

        for testcase in testcases:
            result = self.eval(testcase.func(*testcase.input_data))
            self.assertAlmostEqual(result, testcase.answer, places=1)

    def test_conjgrad(self):
        cgnet = algorithms.ConjugateGradient(
            (10, 5, 1),
            error='binary_crossentropy',
            shuffle_data=True,
            verbose=False,
            update_function='fletcher_reeves',
        )
        x_train, x_test, y_train, y_test = simple_classification()

        cgnet.train(x_train, y_train, x_test, y_test, epochs=50)
        actual_prediction = cgnet.predict(x_test).round().T

        error = metrics.accuracy_score(actual_prediction[0], y_test)
        self.assertAlmostEqual(error, 0.9, places=1)

    def test_compare_bp_and_cg(self):
        x_train, x_test, y_train, y_test = simple_classification()

        compare_networks(
            # Test classes
            partial(
                partial(algorithms.GradientDescent, batch_size='all'),
                step=1.0,
            ),
            partial(
                algorithms.ConjugateGradient,
                update_function='fletcher_reeves'
            ),
            # Test data
            (asfloat(x_train), asfloat(y_train)),
            # Network configurations
            connection=layers.join(
                layers.Input(10),
                layers.Sigmoid(5),
                layers.Sigmoid(1),
            ),
            error='mse',
            shuffle_data=True,
            # Test configurations
            epochs=50,
            show_comparison_plot=False
        )

    def test_conjugate_gradient_fletcher_reeves_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.ConjugateGradient,
                update_function='fletcher_reeves',
                verbose=False,
            ),
            epochs=200,
        )

    def test_conjugate_gradient_dai_yuan_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.ConjugateGradient,
                update_function='dai_yuan',
                verbose=False,
            ),
            epochs=200,
        )

    def test_conjugate_gradient_hentenes_stiefel_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.ConjugateGradient,
                update_function='hentenes_stiefel',
                verbose=False,
            ),
            epochs=500,
        )

    def test_conjugate_gradient_polak_ribiere_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.ConjugateGradient,
                update_function='polak_ribiere',
                verbose=False,
            ),
            epochs=1200,
        )

    def test_conjugate_gradient_liu_storey_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.ConjugateGradient,
                update_function='liu_storey',
                verbose=False,
            ),
            epochs=1200,
        )

    def test_conjgrad_assign_step_exception(self):
        with self.assertRaises(ValueError):
            # Don't have step parameter
            algorithms.ConjugateGradient(
                layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1),
                step=0.01,
            )
