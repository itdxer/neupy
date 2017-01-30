import os

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from neupy import layers, algorithms, environment


environment.reproducible()
environment.speedup()

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
FILES_DIR = os.path.join(CURRENT_DIR, 'files')
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')
TEXT_FILE = os.path.join(FILES_DIR, 'shakespeare.txt')


def load_data(window_size=50, stride=5):
    print("Loading Shakespeare text...")
    with open(TEXT_FILE, 'r') as f:
        text = f.read()

    characters = sorted(list(set(text)))

    n_characters = len(characters)
    n_samples = 1 + ((len(text) - window_size) // stride)

    samples = np.zeros((n_samples, window_size, n_characters))
    targets = np.zeros((n_samples, n_characters))

    for i in tqdm(range(0, len(text) - window_size, stride), total=n_samples):
        sample_id = i // stride
        input_sequence = text[i:i + window_size]

        for j, char in enumerate(input_sequence):
            samples[sample_id, j, characters.index(char)] = 1

        target_char = text[i + window_size]
        targets[sample_id, characters.index(target_char)] = 1

    return train_test_split(samples, targets, train_size=0.9)s


if __name__ == '__main__':
    window_size = 50
    x_train, x_test, y_train, y_test = load_data(window_size, stride=4)
    n_characters = x_train.shape[2]

    network = algorithms.RMSProp(
        [
            layers.Input((window_size, n_characters)),
            layers.LSTM(128, unroll_scan=True),
            layers.Softmax(n_characters),
        ],

        step=0.01,
        verbose=True,
        batch_size=128,
        shuffle_data=True,
        error='categorical_crossentropy',
    )
    network.train(x_train, y_train, x_test, y_test, epochs=10)
