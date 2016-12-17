import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import model_selection, datasets

from neupy import layers, environment, algorithms
from neupy.exceptions import LayerConnectionError
from neupy.utils import theano_random_stream


environment.reproducible()
environment.speedup()


def load_data():
    mnist = datasets.fetch_mldata('MNIST original')

    data = mnist.data / 255.
    data = data - data.mean(axis=0)

    x_train, x_test = model_selection.train_test_split(
        data.astype(np.float32), train_size=(6 / 7.)
    )
    return x_train, x_test


def generate_and_plot_sampels(network, nx, ny):
    space_x = np.linspace(-8, 8, nx)
    space_y = np.linspace(-8, 8, ny)

    plt.figure(figsize=(12, 12))

    grid = gridspec.GridSpec(nx, ny)
    grid.update(wspace=0, hspace=0)

    for i, x in enumerate(space_x):
        for j, y in enumerate(space_y):
            sample = np.array([x, y])
            digit = generator.predict(sample)
            image = digit[0].reshape((28, 28))

            plt.subplot(grid[nx * i + j])
            plt.imshow(image, cmap='Greys_r')
            plt.axis('off')

    plt.show()


class GaussianSample(layers.BaseLayer):
    def validate(self, input_shapes):
        classname = self.__class__.__name__

        if not isinstance(input_shapes, list):
            raise LayerConnectionError("{} layer expected 2 inputs, got 1"
                                       "".format(classname))

        if len(input_shapes) != 2:
            n_inputs = len(input_shapes)
            raise LayerConnectionError("{} layer expected 2 inputs, got {}"
                                       "".format(classname, n_inputs))

        for input_shape in input_shapes:
            ndim = len(input_shape)

            if ndim != 1:
                raise LayerConnectionError("Input layer to {} should be 2D, "
                                           "got {}D".format(classname, ndim))

    @property
    def output_shape(self):
        if self.input_shape:
            return self.input_shape[0]

    def output(self, *input_values):
        mu, sigma = input_values

        random = theano_random_stream()
        return mu + T.exp(sigma) * random.normal(mu.shape)


def vae_loss(expected, predicted):
    x = predicted.owner.inputs[0]

    mean = (encoder > mu).output(x)
    log_var = (encoder > sigma).output(x)

    epsilon = 1e-7
    predicted = T.clip(predicted, epsilon, 1.0 - epsilon)

    crossentropy_loss = T.sum(
        T.nnet.binary_crossentropy(predicted, expected),
        axis=1
    )
    kl_loss = -0.5 * T.sum(
        1 + 2 * log_var - T.square(mean) - T.exp(2 * log_var),
        axis=1
    )

    return (crossentropy_loss + kl_loss).mean()


# Construct Variational Autoencoder
encoder = layers.Input(784) > layers.Tanh(500)

mu = layers.Linear(2, name='mu')
sigma = layers.Linear(2, name='sigma')
sampler = [mu, sigma] > GaussianSample()

decoder = layers.Tanh(500) > layers.Sigmoid(784)

# Train network
network = algorithms.RMSProp(
    encoder > sampler > decoder,

    error=vae_loss,
    batch_size=128,
    shuffle_data=True,
    step=0.001,
    verbose=True,

    decay_rate=0.01,
    addons=[algorithms.WeightDecay],
)

x_train, x_test = load_data()
network.train(x_train, x_train, x_test, x_test, epochs=50)

# Sample digits from the obtained distribution
generator = algorithms.GradientDescent(layers.Input(2) > decoder)
generate_and_plot_sampels(generator, 20, 20)
