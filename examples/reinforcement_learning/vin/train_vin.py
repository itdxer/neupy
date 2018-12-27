import os
import argparse
from functools import partial

import tensorflow as tf

from neupy import layers, init, algorithms, storage
from neupy.utils import (as_tuple, asfloat, flatten, tf_repeat,
                         tensorflow_session)

from loaddata import load_data
from settings import MODELS_DIR, environments
from evaluation import evaluate_accuracy


UNKNOWN = None

parser = argparse.ArgumentParser()
parser.add_argument('--imsize', '-i', choices=[8, 16, 28],
                    type=int, required=True)


def random_weight(shape):
    initializer = init.Normal()
    weight = initializer.sample(shape)
    return tf.Variable(asfloat(weight), dtype=tf.float32)


class ChannelGlobalMaxPooling(layers.BaseLayer):
    @property
    def output_shape(self):
        shape = self.input_shape
        # as_tuple((8, 8), 1) -> (8, 8, 1)
        return as_tuple(shape[:-1], 1)

    def output(self, input_value):
        return tf.reduce_max(input_value, axis=-1, keepdims=True)


class SelectValueAtStatePosition(layers.BaseLayer):
    @property
    def output_shape(self):
        q_output_shape = self.input_shape[0]
        n_filters = q_output_shape[-1]
        return as_tuple(n_filters)  # as_tuple(3) -> (3,)

    def output(self, Q, input_state_1, input_state_2):
        with tf.name_scope("Q-output"):
            # Number of samples depend on the state's batch size.
            # Each iteration we can try to predict direction from
            # multiple different starting points at the same time.
            input_shape = tf.shape(input_state_1)
            n_states = input_shape[1]
            Q_shape = tf.shape(Q)

            indeces = tf.stack([
                # Numer of repetitions depends on the size of
                # the state batch
                tf_repeat(tf.range(Q_shape[0]), n_states),

                # Each state is a coordinate (x and y)
                # that point to some place on a grid.
                tf.cast(flatten(input_state_1), tf.int32),
                tf.cast(flatten(input_state_2), tf.int32),
            ])
            indeces = tf.transpose(indeces, [1, 0])

            # Output is a matrix that has n_samples * n_states rows
            # and n_filters (which is Q.shape[1]) columns.
            return tf.gather_nd(Q, indeces)


def create_VIN(input_image_shape=(8, 8, 2), n_hidden_filters=150,
               n_state_filters=10, k=10):

    SamePadConvolution = partial(layers.Convolution, padding='SAME', bias=None)

    R = layers.join(
        layers.Input(input_image_shape, name='grid-input'),
        layers.Convolution((3, 3, n_hidden_filters),
                           padding='SAME',
                           weight=init.Normal(),
                           bias=init.Normal()),
        SamePadConvolution((1, 1, 1), weight=init.Normal()),
    )

    # Create shared weights
    q_weight = random_weight((3, 3, 1, n_state_filters))
    fb_weight = random_weight((3, 3, 1, n_state_filters))

    Q = R > SamePadConvolution((3, 3, n_state_filters), weight=q_weight)

    for i in range(k):
        V = Q > ChannelGlobalMaxPooling()
        Q = layers.join(
            # Convolve R and V separately and then add outputs together with
            # the Elementwise layer. This part of the code looks different
            # from the one that was used in the original VIN repo, but
            # it does the same operation.
            #
            # conv(x, w) == (conv(x1, w1) + conv(x2, w2))
            # where, x = concat(x1, x2)
            #        w = concat(w1, w2)
            #
            # See code sample from Github Gist: https://bit.ly/2zm3ntN
            [[
                R,
                SamePadConvolution((3, 3, n_state_filters), weight=q_weight)
            ], [
                V,
                SamePadConvolution((3, 3, n_state_filters), weight=fb_weight)
            ]],
            layers.Elementwise(merge_function=tf.add),
        )

    input_state_1 = layers.Input(UNKNOWN, name='state-1-input')
    input_state_2 = layers.Input(UNKNOWN, name='state-2-input')

    # Select the conv-net channels at the state position (S1, S2)
    VIN = [Q, input_state_1, input_state_2] > SelectValueAtStatePosition()

    # Set up softmax layer that predicts actions base on (S1, S2)
    # position. Each action encodes specific direction:
    # N, S, E, W, NE, NW, SE, SW (in the same order)
    VIN = VIN > layers.Softmax(8, bias=None, weight=init.Normal())

    return VIN


def loss_function(expected, predicted):
    epsilon = 1e-7  # for 32-bit float

    predicted = tf.clip_by_value(predicted, epsilon, 1.0 - epsilon)
    expected = tf.cast(flatten(expected), tf.int32)

    log_predicted = tf.log(predicted)
    indeces = tf.stack([tf.range(tf.size(expected)), expected])
    indeces = tf.transpose(indeces, [1, 0])
    errors = tf.gather_nd(log_predicted, indeces)
    return -tf.reduce_mean(errors)


def on_epoch_end_from_steps(steps):
    def on_epoch_end(network):
        if network.last_epoch in steps:
            print("Saving pre-trained VIN model...")
            storage.save(network, env['pretrained_network_file'])

            new_step = steps[network.last_epoch]
            session = tensorflow_session()
            network.variables.step.load(new_step, session)
    return on_epoch_end


if __name__ == '__main__':
    args = parser.parse_args()
    env = environments[args.imsize]

    print("Loading train and test data...")
    x_train, s1_train, s2_train, y_train = load_data(env['train_data_file'])
    x_test, s1_test, s2_test, y_test = load_data(env['test_data_file'])

    print("Initializing VIN...")
    network = algorithms.RMSProp(
        create_VIN(
            env['input_image_shape'],
            n_hidden_filters=150,
            n_state_filters=10,
            k=env['k'],
        ),

        verbose=True,
        error=loss_function,
        signals=on_epoch_end_from_steps(env['steps']),
        **env['training_options']
    )

    print("Training VIN...")
    network.train(
        (x_train, s1_train, s2_train), y_train,
        (x_test, s1_test, s2_test), y_test,
        epochs=env['epochs'],
    )

    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)

    print("Saving pre-trained VIN model...")
    storage.save(network, env['pretrained_network_file'])

    print("Evaluating accuracy on test set...")
    evaluate_accuracy(network.predict, x_test, s1_test, s2_test)
