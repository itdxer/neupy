import numpy as np

from neuralpy import algorithms
from neuralpy.functions import cross_entropy_error
from neuralpy.layers import *

from data import simple_input_train, simple_target_train
from base import BaseTestCase


class QuasiNewtonTestCase(BaseTestCase):
    def setUp(self):
        super(QuasiNewtonTestCase, self).setUp()

        weight1 = np.array([
            [-0.3262846, -0.3899363, -1.31438701, -0.43736622,  0.1234716],
            [-0.31548075, -0.66254391,  0.78722273, -0.51545504, -0.51205823],
            [-0.38036544,  0.34930878,  1.20590571,  0.55030264, -0.94516753],
            [-2.05032326, -0.10582341, -0.33530722,  0.74043659, -0.74645546]
        ])
        weight2 = np.array([
            [-0.25706768,  0.2581464],
            [0.43860057, -0.16620158],
            [-0.87493652,  0.58832669],
            [-1.17300652, -0.21716063],
            [0.66715383, -1.46908589],
            [-1.23662587, -0.85808783]
        ])

        input_layer = SigmoidLayer(3, weight=weight1)
        hidden_layer = SigmoidLayer(5, weight=weight2)

        self.connection = input_layer > hidden_layer > OutputLayer(2)

    def test_quasi_newton(self):
        nw = algorithms.QuasiNewton(
            self.connection,
            step=0.5,
            error=cross_entropy_error,
            use_raw_predict_at_error=False,
            shuffle_data=False,
            update_function='dfp'
        )
        nw.train(simple_input_train, simple_target_train, epochs=150)
        result = np.round(nw.predict(simple_input_train), 3)
        norm = np.linalg.norm(result - simple_target_train)
        self.assertGreater(1e-1, norm)
