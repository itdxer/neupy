from neupy import layers, algorithms

from data import xor_x_train, xor_y_train
from base import BaseTestCase


class SignalsTestCase(BaseTestCase):
    def test_train_epoch_end(self):
        global triggered_times

        triggered_times = 0
        epochs = 4

        def print_message(network):
            global triggered_times
            triggered_times += 1

        network = algorithms.GradientDescent(
            connection=[
                layers.Input(2),
                layers.Sigmoid(2),
                layers.Sigmoid(1)
            ],
            epoch_end_signal=print_message,
            batch_size='all',
        )

        network.train(xor_x_train, xor_y_train, epochs=epochs)
        self.assertEqual(triggered_times, epochs)

    def test_train_end(self):
        global triggered_times

        triggered_times = 0
        epochs = 4

        def print_message(network):
            global triggered_times
            triggered_times += 1

        network = algorithms.GradientDescent(
            connection=[
                layers.Input(2),
                layers.Sigmoid(2),
                layers.Sigmoid(1)
            ],
            train_end_signal=print_message,
            batch_size='all',
        )

        network.train(xor_x_train, xor_y_train, epochs=epochs)
        self.assertEqual(triggered_times, 1)
