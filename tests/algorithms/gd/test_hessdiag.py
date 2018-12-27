from functools import partial

from neupy import algorithms, layers
from neupy import init

from helpers import compare_networks
from helpers import simple_classification
from base import BaseTestCase


class HessianDiagonalTestCase(BaseTestCase):
    def test_hessdiag(self):
        x_train, x_test, y_train, y_test = simple_classification()
        params = dict(
            weight=init.Uniform(-0.1, 0.1),
            bias=init.Uniform(-0.1, 0.1))

        nw = algorithms.HessianDiagonal(
            connection=[
                layers.Input(10),
                layers.Sigmoid(20, **params),
                layers.Sigmoid(1, **params),
            ],
            step=0.1,
            shuffle_data=False,
            verbose=False,
            min_eigval=0.1,
        )
        nw.train(x_train, y_train, epochs=50)
        self.assertGreater(0.2, nw.training_errors[-1])

    def test_compare_bp_and_hessian(self):
        x_train, _, y_train, _ = simple_classification()
        params = dict(
            weight=init.Uniform(-0.1, 0.1),
            bias=init.Uniform(-0.1, 0.1))

        compare_networks(
            # Test classes
            partial(algorithms.GradientDescent, batch_size=None),
            partial(algorithms.HessianDiagonal, min_eigval=0.1),
            # Test data
            (x_train, y_train),
            # Network configurations
            connection=[
                layers.Input(10),
                layers.Sigmoid(20, **params),
                layers.Sigmoid(1, **params),
            ],
            step=0.1,
            shuffle_data=True,
            verbose=False,
            # Test configurations
            epochs=50,
            show_comparison_plot=False
        )

    def test_hessian_diagonal_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.HessianDiagonal,
                verbose=False,
                show_epoch=100,
                step=0.25,
                min_eigval=0.1,
            ),
            epochs=6000,
            min_accepted_error=0.002
        )
