from neupy import layers, algorithms
from neupy.layers.utils import extract_connection, make_one_if_possible

from base import BaseTestCase


class CountParametersTestCase(BaseTestCase):
    def test_count_parameters(self):
        connection = layers.join(
            layers.Input(10),
            layers.Sigmoid(5),
            layers.Sigmoid(2),
        )

        n_parameters = layers.count_parameters(connection)
        self.assertEqual(n_parameters, (10 * 5 + 5) + (5 * 2 + 2))

    def test_count_parameters_single_layer(self):
        hidden_layer = layers.Sigmoid(5)
        layers.join(
            layers.Input(10),
            hidden_layer,
            layers.Sigmoid(2),
        )

        n_parameters = layers.count_parameters(hidden_layer)
        self.assertEqual(n_parameters, 10 * 5 + 5)

    def test_connection_extraction(self):
        connection = layers.Input(2) > layers.Sigmoid(3)
        self.assertIs(extract_connection(connection), connection)

        network = algorithms.GradientDescent(connection)
        self.assertIs(extract_connection(network), connection)

        list_of_layers = [layers.Input(2), layers.Sigmoid(3)]
        actual_connection = extract_connection(list_of_layers)
        self.assertEqual(len(actual_connection), 2)
        self.assertEqual(actual_connection.input_shape, (2,))
        self.assertEqual(actual_connection.output_shape, (3,))

        with self.assertRaisesRegexp(TypeError, "Invalid input type"):
            extract_connection(object)

    def test_make_one_if_possible(self):
        self.assertEqual((3, 4, 5), make_one_if_possible((3, 4, 5)))
        self.assertEqual(10, make_one_if_possible((10,)))
