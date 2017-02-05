import os

import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from neupy import layers, algorithms, environment


environment.reproducible()
environment.speedup()

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
FILES_DIR = os.path.join(CURRENT_DIR, 'files')
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')
TEXT_FILE = os.path.join(FILES_DIR, 'shakespeare.txt')


class TextPreprocessing(object):
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            text = f.read()

        self.text = text
        self.characters = sorted(list(set(text)))
        self.n_characters = len(self.characters)

    def load_samples(self, window_size=50, stride=5):
        text = self.text
        characters = self.characters
        n_characters = self.n_characters
        n_samples = 1 + ((len(text) - window_size) // stride)

        samples = np.zeros((n_samples, window_size, n_characters))
        targets = np.zeros((n_samples, n_characters))

        for i in trange(0, len(text) - window_size, stride, total=n_samples):
            sample_id = i // stride
            input_sequence = text[i:i + window_size]

            for j, char in enumerate(input_sequence):
                samples[sample_id, j, characters.index(char)] = 1

            target_char = text[i + window_size]
            targets[sample_id, characters.index(target_char)] = 1

        return train_test_split(samples, targets, train_size=0.9)

    def encode(self, sequence):
        sequence_size = len(sequence)

        data = np.zeros((sequence_size, self.n_characters))
        data[(np.arange(sequence_size), sequence)] = 1

        return np.expand_dims(data, axis=0)

    def sample(self, predictions, temperature=1.0):
        log_predictions = np.log(predictions) / temperature
        predictions = np.exp(log_predictions)

        predictions = predictions.astype('float64')
        normalized_predictions = predictions / np.sum(predictions)
        probabilities = np.random.multinomial(
            n=1, pvals=normalized_predictions, size=1)

        return np.argmax(probabilities)

    def int_to_string(self, int_sequence):
        char_sequence = [self.characters[i] for i in int_sequence]
        return ''.join(char_sequence)


if __name__ == '__main__':
    window_size = 40

    print("Loading Shakespeare's text ...")
    preprocessor = TextPreprocessing(filepath=TEXT_FILE)
    n_characters = preprocessor.n_characters

    x_train, x_test, y_train, y_test = preprocessor.load_samples(
        window_size, stride=10)

    network = algorithms.RMSProp(
        [
            layers.Input((window_size, n_characters)),
            layers.LSTM(128, unroll_scan=True),
            layers.Softmax(n_characters),
        ],

        step=0.01,
        verbose=True,
        batch_size=128,
        error='categorical_crossentropy',
    )
    network.train(x_train, y_train, x_test, y_test, epochs=10)

    # Number of symbols that will be generated
    n_new_symbols = 1000
    # Which samples to use from the test data
    test_sample_id = 0

    test_sample = x_test[test_sample_id]
    int_sequence = list(test_sample.argmax(axis=1))

    print('\nGenerating new text using pretrained RNN ...')
    for i in trange(n_new_symbols, total=n_new_symbols):
        last_characters = int_sequence[-window_size:]
        data = preprocessor.encode(last_characters)
        output = network.predict(data)
        output = preprocessor.sample(output[0], temperature=0.5)
        int_sequence.append(output)

    print(preprocessor.int_to_string(int_sequence))
