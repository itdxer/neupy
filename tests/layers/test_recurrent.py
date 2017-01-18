import numpy as np
from sklearn.model_selection import train_test_split

from neupy.exceptions import LayerConnectionError
from neupy.datasets import reber
from neupy import layers, algorithms

from base import BaseTestCase


def add_padding(data):
    n_sampels = len(data)
    max_seq_length = max(map(len, data))

    data_matrix = np.zeros((n_sampels, max_seq_length))
    for i, sample in enumerate(data):
        data_matrix[i, -len(sample):] = sample

    return data_matrix


class LSTMTestCase(BaseTestCase):
    def setUp(self):
        super(LSTMTestCase, self).setUp()

        data, labels = reber.make_reber_classification(
            n_samples=100, return_indeces=True)
        data = add_padding(data + 1)  # +1 to shift indeces

        # self.data = x_train, x_test, y_train, y_test
        self.data = train_test_split(data, labels, train_size=0.8)

        self.n_categories = len(reber.avaliable_letters) + 1
        self.n_time_steps = self.data[0].shape[1]

    def test_simple_lstm_sequence_classification(self):
        x_train, x_test, y_train, y_test = self.data

        network = algorithms.RMSProp(
            [
                layers.Input(self.n_time_steps),
                layers.Embedding(self.n_categories, 10),
                layers.LSTM(20),
                layers.Sigmoid(1),
            ],

            step=0.1,
            verbose=False,
            batch_size=64,
            error='binary_crossentropy',
        )
        network.train(x_train, y_train, x_test, y_test, epochs=20)

        y_predicted = network.predict(x_test).round()
        accuracy = (y_predicted.T == y_test).mean()
        self.assertGreater(accuracy, 0.99)

    def test_lstm_connection_exceptions(self):
        with self.assertRaises(LayerConnectionError):
            layers.Input(1) > layers.LSTM(10)
