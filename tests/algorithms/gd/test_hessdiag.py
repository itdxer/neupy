from functools import partial

from neupy import algorithms, layers
from neupy import init

from utils import compare_networks
from data import simple_classification
from base import BaseTestCase


class HessianDiagonalTestCase(BaseTestCase):
    use_sandbox_mode = False

    def test_hessdiag(self):
        x_train, x_test, y_train, y_test = simple_classification()
        nw = algorithms.HessianDiagonal(
            connection=[
                layers.Input(10),
                layers.Sigmoid(20,
                               weight=init.Uniform(-1, 1),
                               bias=init.Uniform(-1, 1)),
                layers.Sigmoid(1,
                               weight=init.Uniform(-1, 1),
                               bias=init.Uniform(-1, 1)),
            ],
            step=0.1,
            shuffle_data=False,
            verbose=False,
            min_eigval=0.01,
        )
        nw.train(x_train / 2, y_train, epochs=10)
        self.assertAlmostEqual(0.10, nw.errors.last(), places=2)

    def test_compare_bp_and_hessian(self):
        x_train, _, y_train, _ = simple_classification()
        compare_networks(
            # Test classes
            algorithms.GradientDescent,
            partial(algorithms.HessianDiagonal, min_eigval=0.01),
            # Test data
            (x_train, y_train),
            # Network configurations
            connection=[
                layers.Input(10),
                layers.Sigmoid(20,
                               weight=init.Uniform(-1, 1),
                               bias=init.Uniform(-1, 1)),
                layers.Sigmoid(1,
                               weight=init.Uniform(-1, 1),
                               bias=init.Uniform(-1, 1)),
            ],
            step=0.1,
            shuffle_data=True,
            verbose=False,
            # Test configurations
            epochs=50,
            show_comparison_plot=False
        )
