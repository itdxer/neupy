from functools import partial

from neupy import algorithms, layers

from helpers import simple_classification, compare_networks
from base import BaseTestCase


class OptimizersTestCase(BaseTestCase):
    def setUp(self):
        super(OptimizersTestCase, self).setUp()
        self.network = layers.join(
            layers.Input(10),
            layers.Sigmoid(20),
            layers.Sigmoid(1),
        )

    def test_adadelta(self):
        x_train, x_test, y_train, y_test = simple_classification()
        optimizer = algorithms.Adadelta(
            self.network,
            batch_size=None,
            verbose=False,
            rho=0.95,
            epsilon=1e-5,
        )
        optimizer.train(x_train, y_train, x_test, y_test, epochs=100)
        self.assertGreater(0.05, optimizer.training_errors[-1])
        self.assertGreater(0.15, optimizer.validation_errors[-1])

    def test_adagrad(self):
        x_train, x_test, y_train, y_test = simple_classification()
        optimizer = algorithms.Adagrad(
            self.network,
            step=0.1,
            batch_size=None,
            verbose=False,
        )
        optimizer.train(x_train, y_train, x_test, y_test, epochs=150)
        self.assertGreater(0.15, optimizer.validation_errors[-1])

    def test_adam(self):
        x_train, x_test, y_train, y_test = simple_classification()
        optimizer = algorithms.Adam(
            self.network,
            step=0.1,
            verbose=False,
            epsilon=1e-4,
            beta1=0.9,
            beta2=0.99,
        )
        optimizer.train(x_train, y_train, x_test, y_test, epochs=200)
        self.assertGreater(0.2, optimizer.validation_errors[-1])

    def test_rmsprop(self):
        x_train, x_test, y_train, y_test = simple_classification()
        optimizer = algorithms.RMSProp(
            self.network,
            step=0.02,
            batch_size=None,
            verbose=False,
            epsilon=1e-5,
            decay=0.9,
        )
        optimizer.train(x_train, y_train, x_test, y_test, epochs=150)
        self.assertGreater(0.15, optimizer.validation_errors[-1])

    def test_momentum(self):
        x_train, x_test, y_train, y_test = simple_classification()
        optimizer = algorithms.Momentum(
            self.network,
            step=0.35,
            momentum=0.99,
            batch_size=None,
            verbose=False,
            nesterov=True,
        )

        optimizer.train(x_train, y_train, x_test, y_test, epochs=30)
        self.assertGreater(0.15, optimizer.validation_errors[-1])

    def test_adamax(self):
        x_train, x_test, y_train, y_test = simple_classification()
        mnet = algorithms.Adamax(
            self.network,
            step=0.1,
            batch_size=None,
            verbose=False,
            epsilon=1e-7,
            beta1=0.9,
            beta2=0.999,
        )
        mnet.train(x_train, y_train, x_test, y_test, epochs=50)
        self.assertGreater(0.15, mnet.training_errors[-1])

    def test_adamax_overfit(self):
        self.assertCanNetworkOverfit(
            partial(algorithms.Adamax, step=0.2, verbose=False),
            epochs=400)


class MomentumTestCase(BaseTestCase):
    def test_momentum_with_minibatch(self):
        x_train, _, y_train, _ = simple_classification()
        compare_networks(
           # Test classes
           partial(algorithms.Momentum, batch_size=None),
           partial(algorithms.Momentum, batch_size=1),
           # Test data
           (x_train, y_train),
           # Network configurations
           network=[
               layers.Input(10),
               layers.Sigmoid(20),
               layers.Sigmoid(1)
           ],
           step=0.25,
           momentum=0.1,
           shuffle_data=True,
           verbose=False,
           # Test configurations
           epochs=40,
           show_comparison_plot=False,
        )

    def test_nesterov_momentum(self):
        x_train, _, y_train, _ = simple_classification()
        compare_networks(
           # Test classes
           partial(algorithms.Momentum, nesterov=False),
           partial(algorithms.Momentum, nesterov=True),
           # Test data
           (x_train, y_train),
           # Network configurations
           network=[
               layers.Input(10),
               layers.Sigmoid(20),
               layers.Sigmoid(1)
           ],
           batch_size=None,
           step=0.25,
           momentum=0.9,
           shuffle_data=True,
           verbose=False,
           # Test configurations
           epochs=10,
           show_comparison_plot=False,
        )
