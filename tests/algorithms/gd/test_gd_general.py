import numpy as np

from neupy import algorithms, layers

from data import simple_classification
from base import BaseTestCase


class BackPropAlgsTestCase(BaseTestCase):
    bp_algorithms = [
        algorithms.GradientDescent,
        algorithms.MinibatchGradientDescent,
        algorithms.ConjugateGradient,
        algorithms.HessianDiagonal,
        algorithms.Hessian,
        algorithms.LevenbergMarquardt,
        algorithms.Momentum,
        algorithms.Quickprop,
        algorithms.QuasiNewton,
        algorithms.RPROP,
        algorithms.IRPROPPlus,
        algorithms.Adadelta,
        algorithms.Adagrad,
        algorithms.Adam,
        algorithms.Adamax,
        algorithms.RMSProp,
    ]

    def test_gd_train_data(self):
        for bp_algorithm_class in self.bp_algorithms:
            self.assertInvalidVectorTrain(
                bp_algorithm_class((2, 1), verbose=False),
                np.array([0, 1]),
                np.array([0]),
                is_feature1d=False
            )

    def test_predict_different_inputs(self):
        for bp_algorithm_class in self.bp_algorithms:
            network = bp_algorithm_class(
                [
                    layers.Input(2),
                    layers.Linear(
                        size=1,
                        bias=np.zeros(1),
                        weight=np.zeros((2, 1))
                    ),
                ],
                verbose=False,
            )
            self.assertInvalidVectorPred(
                network,
                input_vector=np.array([0, 0]),
                target=0,
                is_feature1d=False
            )

    def test_custom_error_functions(self):
        # Test that everything works without fail
        def custom_mse(expected, predicted):
            return (0.5 * (predicted - expected) ** 2).mean()

        x_train, _, y_train, _ = simple_classification()
        gdnet = algorithms.GradientDescent((10, 10, 1), error=custom_mse)
        gdnet.train(x_train, y_train)
