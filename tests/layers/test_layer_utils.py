from neupy import layers

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

    def test_join_empty_connection(self):
        self.assertEqual(layers.join(), None)
