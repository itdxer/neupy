from neupy.algorithms import GradientDescent

from data import xor_input_train, xor_target_train
from base import BaseTestCase


class SignalsTestCase(BaseTestCase):
    def test_train_epoch_end(self):
        global triggered_times

        triggered_times = 0
        epochs = 4

        def print_message(network):
            global triggered_times
            triggered_times += 1

        network = GradientDescent(
            connection=(2, 2, 1),
            epoch_end_signal=print_message,
        )

        network.train(xor_input_train, xor_target_train, epochs=epochs)
        self.assertEqual(triggered_times, epochs)

    def test_train_end(self):
        global triggered_times

        triggered_times = 0
        epochs = 4

        def print_message(network):
            global triggered_times
            triggered_times += 1

        network = GradientDescent(
            connection=(2, 2, 1),
            train_end_signal=print_message,
        )

        network.train(xor_input_train, xor_target_train, epochs=epochs)
        self.assertEqual(triggered_times, 1)
