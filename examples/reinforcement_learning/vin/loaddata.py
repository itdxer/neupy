import pickle
import argparse

import scipy.io
import numpy as np
from sklearn.utils import shuffle
from neupy.utils import asfloat

from settings import environments


def save_data(data, filepath):
    with open(filepath, 'wb') as f:
        # Use protocol 2, for python 2 and 3 compatibility
        return pickle.dump(data, f, protocol=2)


def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--imsize', '-i', choices=[8, 16, 28],
                    type=int, required=True)

if __name__ == '__main__':
    args = parser.parse_args()

    env = environments[args.imsize]
    np.random.seed(args.seed)

    matlab_data = scipy.io.loadmat(env['mat_file'])
    image_data = matlab_data["batch_im_data"]
    image_data = (image_data - 1) / 255.  # obstacles = 1, free zone = 0

    value_data = matlab_data["batch_value_data"]
    s1_data = matlab_data["state_x_data"].astype('int8')
    s2_data = matlab_data["state_y_data"].astype('int8')
    y_data = matlab_data["batch_label_data"].astype('int8')

    image_data = asfloat(image_data.reshape(-1, 1, *env['image_size']))
    value_data = asfloat(value_data.reshape(-1, 1, *env['image_size']))

    x_data = np.append(image_data, value_data, axis=1)
    n_samples = x_data.shape[0]

    training_samples = int(6 / 7.0 * n_samples)

    x_train, x_test = np.split(x_data, [training_samples])
    s1_train, s1_test = np.split(s1_data, [training_samples])
    s2_train, s2_test = np.split(s2_data, [training_samples])
    y_train, y_test = np.split(y_data, [training_samples])

    x_train, s1_train, s2_train, y_train = shuffle(
        x_train, s1_train, s2_train, y_train)

    save_data((x_train, s1_train, s2_train, y_train), env['train_data_file'])
    save_data((x_test, s1_test, s2_test, y_test), env['test_data_file'])
