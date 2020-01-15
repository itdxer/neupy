import math

import numpy as np
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

from neupy.layers import *
from neupy import algorithms, storage
from neupy.architectures.resnet import ResidualBlock
from neupy.utils import asfloat, function_name_scope
from load_cifar10 import read_cifar10


def one_hot_encoder(y_train, y_test):
    y_train, y_test = asfloat(y_train), asfloat(y_test)

    target_scaler = OneHotEncoder(categories='auto', sparse=False)
    y_train = target_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = target_scaler.transform(y_test.reshape(-1, 1))

    return y_train, y_test


@function_name_scope
def ResidualUnit(n_filters, stride=1, has_branch=False, name=''):
    main_branch = join(
        Convolution((3, 3, n_filters), padding='same', stride=stride, bias=None, name='conv' + name + '_branch2a'),
        BatchNorm(name='bn' + name + '_branch2a'),
        Relu(),
        Convolution((3, 3, n_filters), padding='same', bias=None, name='conv' + name + '_branch2b'),
        BatchNorm(name='bn' + name + '_branch2b'),
    )

    if has_branch:
        residual_branch = join(
            Convolution((1, 1, n_filters), stride=stride, bias=None, name='conv' + name + '_branch1'),
            BatchNorm(name='bn' + name + '_branch1'),
        )
    else:
        # Identity defines skip connection, since it doesn't effect the input
        residual_branch = Identity('residual-' + name)

    return join(
        # For the output from two branches we just combine results  with simple elementwise sum operation.
        # The main purpose of the residual connection is to build shortcuts for the gradient during backpropagation.
        (main_branch | residual_branch) >> Elementwise('add', name='add-residual' + name),
        Relu(),
    )


@function_name_scope
def ResidualBlock(n_filters, n_units, name_prefix, stride=1):
    block = ResidualUnit(n_filters, stride=stride, has_branch=True, name=name_prefix)

    for units_index in range(n_units - 1):
        block >>= ResidualUnit(n_filters, name=name_prefix)

    return block


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = read_cifar10()

    x_train, x_test = asfloat(x_train) / 255., asfloat(x_test) / 255.
    mean = np.mean(x_train, axis=0)
    x_train -= mean
    x_test -= mean

    y_train, y_test = one_hot_encoder(y_train, y_test)
    network = join(
        Input((32, 32, 3)),

        ResidualBlock(16, n_units=3, stride=1, name_prefix='1'),
        ResidualBlock(32, n_units=3, stride=2, name_prefix='2'),
        ResidualBlock(64, n_units=3, stride=2, name_prefix='3'),

        GlobalPooling('avg'),
        Softmax(10, name="softmax"),
    )
    optimizer = algorithms.Adam(
        network,
        regularizer=algorithms.l2(
            decay_rate=1e-4,
            include={Convolution: "weight"},
            verbose=True,
        ),
        step=0.001,
        # step=algorithms.step_decay(
        #     initial_value=0.001,
        #     # Parameter controls step redution frequency. The larger
        #     # the value the slower step parameter decreases. Step will
        #     # be reduced after every mini-batch update. In the training
        #     # data we have 500 mini-batches.
        #     reduction_freq=5 * 500,
        # ),
        batch_size=32,
        loss='categorical_crossentropy',
        shuffle_data=True,
        verbose=True,
    )
    storage.save_pickle(network, "models/model.pickle")

    for epoch in range(30):
        optimizer.train(x_train, y_train, batch_size=32, epochs=1)
        storage.save_pickle(network, "models/model-{:0>2}.pickle".format(epoch))
        y_predict = optimizer.predict(x_test)
        print('test mean:', np.mean(y_predict.argmax(axis=1) == y_test.argmax(axis=1)))

    y_predicted = optimizer.predict(x_test).argmax(axis=1)
    y_test_labels = np.asarray(y_test.argmax(axis=1)).reshape(len(y_test))

    print(metrics.classification_report(y_test_labels, y_predicted))
    score = metrics.accuracy_score(y_test_labels, y_predicted)
    print("Validation accuracy: {:.2%}".format(score))
    print(metrics.confusion_matrix(y_predicted, y_test_labels))
