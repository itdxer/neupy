import os
import math
import random
import argparse

import numpy as np
from scipy.io import wavfile


current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'data')
music_dir = os.path.join(data_dir, 'music_wav')
speech_dir = os.path.join(data_dir, 'speech_wav')

splited_data_file = os.path.join(data_dir, 'splited_data.npz')

train_size = 0.85

SPEECH = 0
MUSIC = 1

parser = argparse.ArgumentParser()
parser.add_argument('--seed', '-s', default=None, dest='seed',
                    help="This parameter makes results reproduceble",
                    type=int)


def train_test_data():
    data = np.load(splited_data_file)
    return data['x_train'], data['x_test'], data['y_train'], data['y_test']


if __name__ == '__main__':
    print("Start read data")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    x_train, x_test = [], []
    y_train, y_test = [], []

    for class_code, directory in enumerate([music_dir, speech_dir]):
        filenames = os.listdir(directory)
        n_train_samples = math.floor(len(filenames) * train_size)
        train_filenames = random.sample(filenames, k=n_train_samples)

        for filename in filenames:
            full_filepath = os.path.join(directory, filename)
            _, wav_vector = wavfile.read(full_filepath)

            if filename in train_filenames:
                x_train.append(wav_vector)
            else:
                x_test.append(wav_vector)

        classes = np.repeat(class_code, len(filenames))
        y_train = np.concatenate([y_train, classes[:n_train_samples]])
        y_test = np.concatenate([y_test, classes[n_train_samples:]])

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    print("Train data shape: {}".format(x_train.shape))
    print("Test data shape: {}".format(x_test.shape))

    print("Save data in file")
    np.savez(splited_data_file, x_train=x_train, x_test=x_test,
             y_train=y_train, y_test=y_test)
