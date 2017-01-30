import os
import argparse
from functools import partial

import theano
import theano.tensor as T
import numpy as np

from neupy.utils import as_tuple, asfloat, asint
from neupy import layers, init, algorithms, environment, storage

from loaddata import load_data
from settings import MODELS_DIR, environments
from evaluation import evaluate_accuracy


parser = argparse.ArgumentParser()
parser.add_argument('--imsize', '-i', choices=[8, 16, 28],
                    type=int, required=True)


def random_weight(shape):
    weight = 0.01 * np.random.random(shape)
    return theano.shared(asfloat(weight))


class GlobalMaxPooling(layers.BaseLayer):
    @property
    def output_shape(self):
        shape = self.input_shape
        # as_tuple(1, (8, 8)) -> (1, 8, 8)
        return as_tuple(1, shape[1:])

    def output(self, input_value):
        return T.max(input_value, axis=1, keepdims=True)


class SelectValueAtStatePosition(layers.BaseLayer):
    @property
    def output_shape(self):
        q_output_shape = self.input_shape[0]
        n_filters = q_output_shape[0]
        # as_tuple(3) -> (3,)
        return as_tuple(n_filters)

    def output(self, Q, input_state_1, input_state_2):
        # Number of samples dependce on the state batch size.
        # Each iteration we can try to predict direction from
        # multiple different starting points at the same time.
        n_states = input_state_1.shape[1]

        # Output is a matrix that has n_samples * n_states rows
        # and n_filters (which is Q.shape[1]) columns.
        return Q[
            # Numer of repetitions depends on the size of
            # the state batch
            T.extra_ops.repeat(T.arange(Q.shape[0]), n_states),

            # Extract all channels
            :,

            # Each state is a coordinate (x and y)
            # that point to some place on a grid.
            asint(input_state_1.flatten()),
            asint(input_state_2.flatten()),
        ]


def create_VIN(input_image_shape=(2, 8, 8), n_hidden_filters=150,
               n_state_filters=10, k=10):

    HalfPaddingConv = partial(layers.Convolution, padding='half', bias=None)

    R = layers.join(
        layers.Input(input_image_shape, name='grid-input'),
        layers.Convolution((n_hidden_filters, 3, 3),
                           padding='half',
                           weight=init.Normal(),
                           bias=init.Normal()),
        HalfPaddingConv((1, 1, 1), weight=init.Normal()),
    )

    # Create shared weights
    q_weight = random_weight((n_state_filters, 1, 3, 3))
    fb_weight = random_weight((n_state_filters, 1, 3, 3))

    Q = R > HalfPaddingConv((n_state_filters, 3, 3), weight=q_weight)

    for i in range(k):
        V = Q > GlobalMaxPooling()
        Q = layers.join(
            # Convolve R and V separately and then add
            # outputs together with the Elementwise layer
            [[
                R,
                HalfPaddingConv((n_state_filters, 3, 3), weight=q_weight)
            ], [
                V,
                HalfPaddingConv((n_state_filters, 3, 3), weight=fb_weight)
            ]],
            layers.Elementwise(merge_function=T.add),
        )

    input_state_1 = layers.Input(10, name='state-1-input')
    input_state_2 = layers.Input(10, name='state-2-input')

    # Select the conv-net channels at the state position (S1, S2)
    VIN = [Q, input_state_1, input_state_2] > SelectValueAtStatePosition()

    # Set up softmax layer that predicts actions base on (S1, S2)
    # position. Each action encodes specific direction:
    # N, S, E, W, NE, NW, SE, SW (in the same order)
    VIN = VIN > layers.Softmax(8, bias=None, weight=init.Normal())

    return VIN


def loss_function(expected, predicted):
    epsilon = 1e-7
    log_predicted = T.log(T.clip(predicted, epsilon, 1.0 - epsilon))
    errors = log_predicted[T.arange(expected.size), asint(expected.flatten())]
    return -T.mean(errors)


def on_epoch_end(network):
    steps = {
        30: 0.005,
        60: 0.002,
        90: 0.001,
    }

    if network.last_epoch in steps:
        new_step = steps[network.last_epoch]
        network.variables.step.set_value(new_step)


if __name__ == '__main__':
    environment.speedup()

    args = parser.parse_args()
    env = environments[args.imsize]

    x_train, s1_train, s2_train, y_train = load_data(env['train_data_file'])
    x_test, s1_test, s2_test, y_test = load_data(env['test_data_file'])

    network = algorithms.RMSProp(
        create_VIN(
            env['input_image_shape'],
            n_hidden_filters=150,
            n_state_filters=10,
            k=env['k'],
        ),

        step=0.01,
        verbose=True,
        batch_size=12,
        error=loss_function,
        epoch_end_signal=on_epoch_end,

        decay=0.9,
        epsilon=1e-6,
    )
    network.train((x_train, s1_train, s2_train), y_train,
                  (x_test, s1_test, s2_test), y_test,
                  epochs=120)

    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)

    storage.save(network, env['pretrained_network_file'])
    evaluate_accuracy(network.connection.compile(), x_test, s1_test, s2_test)
