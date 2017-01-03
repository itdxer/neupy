import theano.tensor as T

from neupy import layers, algorithms
from neupy.exceptions import InvalidConnection
from neupy.algorithms.constructor import (create_output_variable,
                                          ConstructibleNetwork,
                                          generate_layers)

from base import BaseTestCase
from data import simple_classification


class NetworkConstructorTestCase(BaseTestCase):
    def test_generate_layers(self):
        network = generate_layers([1, 2, 3])

        layer_types = (layers.Input, layers.Sigmoid, layers.Sigmoid)
        output_shapes = [(1,), (2,), (3,)]

        for layer, layer_type in zip(network, layer_types):
            self.assertIsInstance(layer, layer_type)

        for layer, output_shape in zip(network, output_shapes):
            self.assertEqual(layer.output_shape, output_shape)

    def test_generate_layers_expcetion(self):
        with self.assertRaises(ValueError):
            generate_layers((5,))

    def test_custom_output_variable(self):
        def error_func(expected, predicted):
            return T.abs(expected - predicted)

        error_func.expected_dtype = T.vector
        var = create_output_variable(error_func, name='test_func')

        self.assertIn('vector', str(var.type))


class ConstructibleNetworkTestCase(BaseTestCase):
    def test_training_with_multiple_inputs(self):
        network = algorithms.GradientDescent(
            [
                [
                    layers.Input(2) > layers.Sigmoid(3),
                    layers.Input(3) > layers.Sigmoid(5),
                ],
                layers.Concatenate(),
                layers.Sigmoid(1),
            ],
            step=0.1,
            verbose=False
        )

        x_train, x_test, y_train, y_test = simple_classification(
            n_samples=100, n_features=5)

        x_train_2, x_train_3 = x_train[:, :2], x_train[:, 2:]
        x_test_2, x_test_3 = x_test[:, :2], x_test[:, 2:]

        network.train([x_train_2, x_train_3], y_train,
                      [x_test_2, x_test_3], y_test,
                      epochs=100)

        error = network.validation_errors[-1]
        self.assertAlmostEqual(error, 0.14, places=2)

    def test_multi_output_exception(self):
        connection = layers.Input(10) > [
            [layers.Sigmoid(20)],
            [layers.Sigmoid(30)],
        ]

        with self.assertRaises(InvalidConnection):
            ConstructibleNetwork(connection)

    def test_no_updates_by_default(self):
        net = ConstructibleNetwork(layers.Input(10) > layers.Sigmoid(1))
        updates = net.init_param_updates(layer=None, parameter=None)
        self.assertEqual(updates, [])
