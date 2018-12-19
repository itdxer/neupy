import numpy as np
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

from neupy.layers import *
from neupy import algorithms
from neupy.utils import asfloat
from load_cifar10 import read_cifar10


def process_cifar10_data(x_train, x_test):
    x_train, x_test = asfloat(x_train), asfloat(x_test)

    mean = x_train.mean(axis=(0, 1, 2)).reshape(1, 1, 1, -1)
    std = x_train.std(axis=(0, 1, 2)).reshape(1, 1, 1, -1)

    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std

    return x_train, x_test


def one_hot_encoder(y_train, y_test):
    y_train, y_test = asfloat(y_train), asfloat(y_test)

    target_scaler = OneHotEncoder(categories='auto', sparse=False)
    y_train = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = target_scaler.transform(y_test.reshape(-1, 1))

    return y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = read_cifar10()

    x_train, x_test = process_cifar10_data(x_train, x_test)
    y_train, y_test = one_hot_encoder(y_train, y_test)

    network = algorithms.Adam(
        [
            Input((32, 32, 3)),

            Convolution((3, 3, 32)) > Relu(),
            Convolution((3, 3, 32)) > Relu(),
            MaxPooling((2, 2)),
            Dropout(0.2),

            Convolution((3, 3, 64)) > Relu(),
            Convolution((3, 3, 64)) > Relu(),
            MaxPooling((2, 2)),
            Dropout(0.2),

            Reshape(),
            Relu(512) > Dropout(0.5),
            Softmax(10),
        ],

        step=algorithms.step_decay(
            initial_value=0.001,
            # Parameter controls step redution frequency. The larger
            # the value the slower step parameter decreases. Step will
            # be reduced after every mini-batch update. In the training
            # data we have 500 mini-batches.
            reduction_freq=5 * 500,
        ),
        regularizer=algorithms.l2(0.00001),

        error='categorical_crossentropy',
        batch_size=100,
        shuffle_data=True,
        verbose=True,
    )
    network.train(x_train, y_train, x_test, y_test, epochs=30)

    y_predicted = network.predict(x_test).argmax(axis=1)
    y_test_labels = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))

    print(metrics.classification_report(y_test_labels, y_predicted))
    score = metrics.accuracy_score(y_test_labels, y_predicted)
    print("Validation accuracy: {:.2%}".format(score))
    print(metrics.confusion_matrix(y_predicted, y_test_labels))
