import pickle
from functools import partial

from neupy import algorithms, layers
from neupy.exceptions import InvalidConnection

from base import BaseTestCase
from helpers import simple_classification


class GradientDescentTestCase(BaseTestCase):
    def test_gd(self):
        x_train, _, y_train, _ = simple_classification()

        network = algorithms.GradientDescent(
            layers.Input(10) > layers.Tanh(20) > layers.Tanh(1),
            step=0.1,
            verbose=False
        )
        network.train(x_train, y_train, epochs=100)
        self.assertLess(network.training_errors[-1], 0.05)

    def test_gd_get_params_method(self):
        network = algorithms.GradientDescent(
            layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1))

        self.assertIn(
            'connection',
            network.get_params(with_connection=True),
        )
        self.assertNotIn(
            'connection',
            network.get_params(with_connection=False),
        )

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

    def test_small_network_representation(self):
        network = algorithms.GradientDescent(
            layers.Input(2) > layers.Sigmoid(3) > layers.Sigmoid(1))
        self.assertIn("Input(2) > Sigmoid(3) > Sigmoid(1)", str(network))

    def test_large_network_representation(self):
        network = algorithms.GradientDescent([
            layers.Input(1),
            layers.Sigmoid(1),
            layers.Sigmoid(1),
            layers.Sigmoid(1),
            layers.Sigmoid(1),
            layers.Sigmoid(1),
        ])
        self.assertIn("[... 6 layers ...]", str(network))

    def test_raise_exception_for_multioutputs(self):
        output_1 = layers.Relu(1)
        output_2 = layers.Relu(2)
        network = layers.Input(5) > [output_1, output_2]

        error_message = "should have one output layer"
        with self.assertRaisesRegexp(InvalidConnection, error_message):
            algorithms.GradientDescent(network)

    def test_gd_storage(self):
        network = algorithms.GradientDescent([
                layers.Input(2),
                layers.Sigmoid(3),
                layers.Sigmoid(1),
            ],
            step=0.2,
            shuffle_data=True,
        )
        recovered_network = pickle.loads(pickle.dumps(network))

        self.assertAlmostEqual(self.eval(recovered_network.step), 0.2)
        self.assertEqual(recovered_network.shuffle_data, True)
