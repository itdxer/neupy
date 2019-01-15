import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import model_selection, datasets

from neupy import algorithms
from neupy.layers import *
from neupy.exceptions import LayerConnectionError


def load_data():
    X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

    X = X / 255.
    X -= X.mean(axis=0)

    x_train, x_test = model_selection.train_test_split(
        X.astype(np.float32),
        test_size=(1 / 7.)
    )
    return x_train, x_test


def generate_and_plot_sampels(network, nx, ny):
    space_x = np.linspace(-3, 3, nx)
    space_y = np.linspace(-3, 3, ny)

    plt.figure(figsize=(12, 12))

    grid = gridspec.GridSpec(nx, ny)
    grid.update(wspace=0, hspace=0)

    for i, x in enumerate(space_x):
        for j, y in enumerate(space_y):
            sample = np.array([[x, y]])
            digit = generator.predict(sample)
            image = digit[0].reshape((28, 28))

            plt.subplot(grid[nx * i + j])
            plt.imshow(image, cmap='Greys_r')
            plt.axis('off')

    plt.show()


class GaussianSample(BaseLayer):
    def get_output_shape(self, mu_shape, sigma_shape):
        return mu_shape

    def output(self, mu, sigma, **kwargs):
        return mu + tf.exp(sigma) * tf.random_normal(tf.shape(mu))


class Collect(Identity):
    def output(self, input, **kwargs):
        tf.add_to_collection(self.name, input)
        return input


def binary_crossentropy(expected, predicted):
    epsilon = 1e-7  # smallest positive 32-bit float number
    predicted = tf.clip_by_value(predicted, epsilon, 1.0 - epsilon)

    return tf.reduce_sum(
        expected * tf.log(predicted) +
        (1 - expected) * tf.log(1 - predicted),
        axis=1
    )


def vae_loss(expected, predicted):
    mean = tf.get_collection('mu')[-1]
    log_var = tf.get_collection('sigma')[-1]

    epsilon = 1e-7
    predicted = tf.clip_by_value(predicted, epsilon, 1.0 - epsilon)

    crossentropy_loss = binary_crossentropy(expected, predicted)
    kl_loss = tf.reduce_sum(
        1 + log_var - tf.square(mean) - tf.exp(log_var),
        axis=1
    )
    return tf.reduce_mean(-crossentropy_loss - 0.5 * kl_loss)


if __name__ == '__main__':
    # Construct Variational Autoencoder
    network = join(
        # Encoder
        Input(784, name='input'),
        Tanh(256),

        # Sampler
        parallel(
            # Two is the maximum number of dimensions that we can visualize
            Linear(2, name='mu') >> Collect('mu'),
            Linear(2, name='sigma') >> Collect('sigma'),
        ),
        GaussianSample(),

        # Decoder
        # Note: Identity layer acts as a reference. Using it
        # we can easily cut the decoder from the network
        Identity('decoder'),
        Tanh(256),
        Sigmoid(784),
    )

    # Train network
    optimizer = algorithms.RMSProp(
        network,
        loss=vae_loss,
        regularizer=algorithms.l2(0.001),

        batch_size=128,
        shuffle_data=True,
        step=0.001,
        verbose=True,
    )

    x_train, x_test = load_data()
    optimizer.train(x_train, x_train, x_test, x_test, epochs=50)

    # Sample digits from the obtained distribution
    generator = Input(2) >> network.start('decoder')
    generate_and_plot_sampels(generator, 15, 15)
