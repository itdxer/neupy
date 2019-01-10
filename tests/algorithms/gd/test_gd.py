import pickle
from functools import partial

from neupy import algorithms, layers
from neupy.exceptions import InvalidConnection

from base import BaseTestCase


class GradientDescentTestCase(BaseTestCase):
    def test_gd_overfit(self):
        self.assertCanNetworkOverfit(
            partial(algorithms.GradientDescent, step=1.0, verbose=False),
            epochs=4000,
        )

    def test_gd_minibatch_overfit(self):
        self.assertCanNetworkOverfit(
            partial(
                algorithms.GradientDescent,
                step=0.5,
                batch_size=5,
                verbose=False,
            ),
            epochs=4000,
        )

    def test_large_network_representation(self):
        optimizer = algorithms.GradientDescent([
            layers.Input(1),
            layers.Sigmoid(1),
            layers.Sigmoid(1),
            layers.Sigmoid(1),
            layers.Sigmoid(1),
            layers.Sigmoid(2),
        ])
        self.assertIn(
            "(?, 1) -> [... 6 layers ...] -> (?, 2)",
            str(optimizer))

    def test_raise_exception_for_multioutputs(self):
        network = layers.join(
            layers.Input(5),
            layers.parallel(
                layers.Relu(1),
                layers.Relu(2),
            )
        )
        error_message = "should have one output layer"
        with self.assertRaisesRegexp(InvalidConnection, error_message):
            algorithms.GradientDescent(network)

    def test_network_initializations(self):
        possible_networks = (
            # as a list
            [layers.Input(2), layers.Sigmoid(3), layers.Tanh(1)],

            # as forward sequence with inline operators
            layers.Input(2) > layers.Relu(10) > layers.Tanh(1),
            layers.Input(2) >> layers.Relu(10) >> layers.Tanh(1),
        )

        for i, network in enumerate(possible_networks, start=1):
            optimizer = algorithms.GradientDescent(network)
            message = "[Test #{}] Network: {}".format(i, network)
            self.assertEqual(len(optimizer.network.layers), 3, msg=message)

    def test_gd_get_params_method(self):
        optimizer = algorithms.GradientDescent([
            layers.Input(2),
            layers.Sigmoid(3),
            layers.Sigmoid(1),
        ])

        self.assertIn(
            'network',
            optimizer.get_params(with_network=True),
        )
        self.assertNotIn(
            'network',
            optimizer.get_params(with_network=False),
        )

    def test_gd_storage(self):
        optimizer = algorithms.GradientDescent(
            [
                layers.Input(2),
                layers.Sigmoid(3),
                layers.Sigmoid(1),
            ],
            step=0.2,
            shuffle_data=True,
        )
        recovered_optimizer = pickle.loads(pickle.dumps(optimizer))

        self.assertAlmostEqual(self.eval(recovered_optimizer.step), 0.2)
        self.assertEqual(recovered_optimizer.shuffle_data, True)
