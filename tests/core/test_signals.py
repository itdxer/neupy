from neupy.algorithms import Backpropagation

from data import xor_input_train, xor_target_train
from base import BaseTestCase


class SignalsTestCase(BaseTestCase):
    def test_train_state(self):
        global triggered_times
        triggered_times = 0
        epochs = 4

        def print_message(network):
            global triggered_times
            triggered_times += 1

        def print_message2(network):
            global triggered_times
            triggered_times += 1

        network = Backpropagation(
            connection=(2, 2, 1),
            train_epoch_end_signal=print_message,
            train_end_signal=print_message2,
        )
        network.train(xor_input_train, xor_target_train, epochs=epochs)

        self.assertEqual(triggered_times, epochs + 1)
