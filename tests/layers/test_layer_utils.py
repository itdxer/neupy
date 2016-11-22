from neupy import layers

from base import BaseTestCase


class CountParametersTestCase(BaseTestCase):
    def test_count_parameters(self):
        connection = layers.join(
            layers.Input(10),
            layers.Sigmoid(5),
            layers.Sigmoid(2),
        )
        connection.initialize()

        n_parameters = layers.count_parameters(connection)
        self.assertEqual(n_parameters, (10 * 5 + 5) + (5 * 2 + 2))
