import numpy as np
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

from neupy.layers import *
from neupy import algorithms
from neupy.utils import asfloat
from load_cifar10 import read_cifar10


def process_cifar10_data(x_train, x_test):
    x_train, x_test = asfloat(x_train), asfloat(x_test)

    mean = x_train.mean(axis=(0, 2, 3)).reshape((1, -1, 1, 1))
    std = x_train.std(axis=(0, 2, 3)).reshape((1, -1, 1, 1))

    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std

    return x_train, x_test


def one_hot_encoder(y_train, y_test):
    y_train, y_test = asfloat(y_train), asfloat(y_test)

    target_scaler = OneHotEncoder(categories='auto')
    y_train = target_scaler.fit_transform(y_train.reshape((-1, 1))).todense()
    y_test = target_scaler.transform(y_test.reshape((-1, 1))).todense()

    return y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = read_cifar10()

    x_train, x_test = process_cifar10_data(x_train, x_test)
    y_train, y_test = one_hot_encoder(y_train, y_test)

    network = algorithms.Adadelta(
        [
            Input((3, 32, 32)),

            Convolution((64, 3, 3)) > PRelu() > BatchNorm(),
            Convolution((64, 3, 3)) > PRelu() > BatchNorm(),
            MaxPooling((2, 2)),

            Convolution((128, 3, 3)) > PRelu() > BatchNorm(),
            Convolution((128, 3, 3)) > PRelu() > BatchNorm(),
            MaxPooling((2, 2)),

            Convolution((256, 3, 3)) > PRelu() > BatchNorm(),
            Reshape(),

            Relu(512) > Dropout(0.5),
            Relu(256) > Dropout(0.5),
            Softmax(10),
        ],

        error='categorical_crossentropy',
        step=0.2,
        shuffle_data=True,
        batch_size=100,
        verbose=True,

        # Parameter controls step redution frequency. The larger
        # the value the slower step parameter decreases.
        # Step will be reduced after every mini-batch update. In the
        # training data we have 500 mini-batches.
        reduction_freq=4 * 500,
        # The larger the value the more impact regularization
        # makes on the parameter training
        # decay_rate=0.1,
        addons=[algorithms.StepDecay],
    )
    network.architecture()
    network.train(x_train, y_train, x_test, y_test, epochs=20)

    y_predicted = network.predict(x_test).argmax(axis=1)
    y_test_labels = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))

    print(metrics.classification_report(y_test_labels, y_predicted))
    score = metrics.accuracy_score(y_test_labels, y_predicted)
    print("Validation accuracy: {:.2%}".format(score))
    print(metrics.confusion_matrix(y_predicted, y_test_labels))
