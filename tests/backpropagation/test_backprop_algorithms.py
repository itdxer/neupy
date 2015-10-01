import numpy as np

from neupy import algorithms, layers

from base import BaseTestCase


class BackPropAlgsTestCase(BaseTestCase):
    bp_algorithms = [
        algorithms.Backpropagation,
        algorithms.ConjugateGradient,
        algorithms.MinibatchGradientDescent,
        algorithms.HessianDiagonal,
        algorithms.LevenbergMarquardt,
        algorithms.Momentum,
        algorithms.QuasiNewton,
        algorithms.Quickprop,
        algorithms.RPROP,
        algorithms.IRPROPPlus,
    ]

    def test_train_data(self):
        for bp_algorithm_class in self.bp_algorithms:
            self.assertInvalidVectorTrain(
                bp_algorithm_class((2, 1), verbose=False),
                np.array([0, 1]),
                np.array([0]),
                row1d=True
            )

    def test_predict_different_inputs(self):
        for bp_algorithm_class in self.bp_algorithms:
            network = bp_algorithm_class(
                layers.LinearLayer(2) > layers.OutputLayer(1),
                verbose=False,
                use_bias=False
            )
            self.assertInvalidVectorPred(network, np.array([0, 0]), 0,
                                         row1d=True)
