import os
import pickle

import theano
import scipy.io
import numpy as np
from sklearn.utils import shuffle
from neupy.utils import asfloat


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
MAT_DATA = os.path.join(DATA_DIR, 'gridworld_8.mat')
TRAIN_DATA = os.path.join(DATA_DIR, 'gridworld-8-train.pickle')
TEST_DATA = os.path.join(DATA_DIR, 'gridworld-8-test.pickle')


def save_data(data, filepath):
    with open(filepath, 'wb') as f:
        return pickle.dump(data, f)


def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    np.random.seed(0)

    im_size = (8, 8)

    matlab_data = scipy.io.loadmat(MAT_DATA)
    image_data = matlab_data["batch_im_data"]
    image_data = (image_data - 1) / 255.  # obstacles = 1, free zone = 0

    value_data = matlab_data["batch_value_data"]
    s1_data = matlab_data["state_x_data"].astype('int8')
    s2_data = matlab_data["state_y_data"].astype('int8')
    y_data = matlab_data["batch_label_data"].astype('int8')

    image_data = asfloat(image_data.reshape(-1, 1, *im_size))
    value_data = asfloat(value_data.reshape(-1, 1, *im_size))
    x_data = np.append(image_data, value_data, axis=1)
    n_samples = x_data.shape[0]

    training_samples = int(6 / 7.0 * n_samples)

    x_train, x_test = np.split(x_data, [training_samples])
    s1_train, s1_test = np.split(s1_data, [training_samples])
    s2_train, s2_test = np.split(s2_data, [training_samples])
    y_train, y_test = np.split(y_data, [training_samples])

    x_train, s1_train, s2_train, y_train = shuffle(
        x_train, s1_train, s2_train, y_train)

    save_data((x_train, s1_train, s2_train, y_train), TRAIN_DATA)
    save_data((x_test, s1_test, s2_test, y_test), TEST_DATA)
