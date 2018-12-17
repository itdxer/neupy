import os
import tarfile

import numpy as np
import six
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

from imagenet_tools import FILES_DIR


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_TAR_PATH = os.path.join(FILES_DIR, "cifar-10-python.tar.gz")
CIFAR10_PATH = os.path.join(FILES_DIR, "cifar-10-batches-py")

TEST_FILE = 'test_batch'
TRAINING_FILES = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                  'data_batch_4', 'data_batch_5']


def download_cifar10_if_not_found():
    if os.path.exists(CIFAR10_PATH):
        print("CIFAR10 was already downloaded and extracted.")
        print("  Output path: {}".format(CIFAR10_PATH))
        print("")
        return

    print("Downloading CIFAR 10 dataset...")
    print("  From URL:    {}".format(CIFAR10_URL))
    print("  Output path: {}".format(CIFAR10_TAR_PATH))
    urlretrieve(CIFAR10_URL, CIFAR10_TAR_PATH)
    print("Downloading finished\n")

    print("Extracting CIFAR 10 dataset...")
    with tarfile.open(CIFAR10_TAR_PATH) as tar:
        tar.extractall(path=FILES_DIR)

    print("Extracting finished\n")


def read_cifar10_file(filename):
    path = os.path.join(CIFAR10_PATH, filename)

    with open(path, 'rb') as f:
        # Specify encoding for python 3 in order to be able to
        # read files that has been created in python 2
        options = {'encoding': 'latin1'} if six.PY3 else {}
        batch = pickle.load(f, **options)

    return batch['data'], batch['labels']


def read_cifar10():
    download_cifar10_if_not_found()

    print("Reading CIFAR10 data...")
    x_train, y_train = [], []

    for training_filename in TRAINING_FILES:
        data, labels = read_cifar10_file(training_filename)
        x_train.append(data)
        y_train.append(labels)

    x_test, y_test = read_cifar10_file(TEST_FILE)

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_train = x_train.reshape((x_train.shape[0], 3, 32, 32))
    x_train = np.transpose(x_train, (0, 2, 3, 1))

    x_test = x_test.reshape((x_test.shape[0], 3, 32, 32))
    x_test = np.transpose(x_test, (0, 2, 3, 1))

    print("Finished reading CIFAR10 data\n")
    return x_train, x_test, y_train, y_test
