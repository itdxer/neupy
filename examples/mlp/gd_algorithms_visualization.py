from functools import partial

import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from neupy.utils import asfloat
from neupy import algorithms, layers, environment


environment.reproducible()

input_data = np.array([
    [0.9, 0.3],
    [0.5, 0.3],
    [0.2, 0.1],
    [0.7, 0.5],
    [0.1, 0.8],
    [0.1, 0.9],
])
target_data = np.array([
    [1],
    [1],
    [1],
    [0],
    [0],
    [0],
])
default_weight = np.array([[-4.], [-4.]])

weights = None
current_epoch = 0


def draw_countour(xgrid, ygrid, target_function):
    output = np.zeros((xgrid.shape[0], ygrid.shape[0]))

    for i, x in enumerate(xgrid):
        for j, y in enumerate(ygrid):
            output[j, i] = target_function(x, y)

    X, Y = np.meshgrid(xgrid, ygrid)

    plt.contourf(X, Y, output, 20, alpha=1, cmap='Blues')
    plt.colorbar()


def weight_quiver(weights, color='c'):
    plt.quiver(weights[0, :-1],
               weights[1, :-1],
               weights[0, 1:] - weights[0, :-1],
               weights[1, 1:] - weights[1, :-1],
               scale_units='xy', angles='xy', scale=1,
               color=color)


class NoBiasSigmoid(layers.Sigmoid):
    def output(self, input_value):
        # Miltiply bias by zero to disable it. We need to include it in
        # formula, because Theano use update rules for it.
        summated = T.dot(input_value, self.weight) + (0 * self.bias)
        return self.activation_function(summated)


def copy_weight(weight):
    return weight.get_value().copy()


def save_epoch_weight(net):
    """
    Signal processor which save weight update for every
    epoch.
    """
    global weights
    global current_epoch

    input_layer_weight = copy_weight(net.layers[1].weight)
    weights[:, current_epoch + 1:current_epoch + 2] = input_layer_weight


def get_connection():
    """
    Generate new connections every time when we call it.
    """
    input_layer = layers.Input(2)
    output_layer = NoBiasSigmoid(1, weight=default_weight.copy())
    return input_layer > output_layer


def draw_quiver(network_class, name, color='r'):
    """
    Train algorithm and draw quiver for every epoch update
    for this algorithm.
    """
    global weights
    global current_epoch

    bpn = network_class(
        get_connection(),
        step=0.3,
        epoch_end_signal=save_epoch_weight
    )

    # We don't know in advance number of epochs that network
    # need to reach the goal. For this reason we use 1000 as
    # an upper limit for all network epochs, later we
    # need to fix
    weights = np.zeros((2, 1000))
    weights[:, 0:1] = default_weight.copy()

    current_epoch = 0
    while bpn.prediction_error(input_data, target_data) > 0.125:
        bpn.train(input_data, target_data, epochs=1)
        current_epoch += 1

    weights = weights[:, :current_epoch + 1]
    weight_quiver(weights, color=color)

    label = "{name} ({n} steps)".format(name=name, n=current_epoch)
    return mpatches.Patch(color=color, label=label)


def target_function(network, x, y):
    weight = network.layers[1].weight
    new_weight = np.array([[x], [y]])
    weight.set_value(asfloat(new_weight))
    return network.prediction_error(input_data, target_data)


# Get data for countour plot
bp_network = algorithms.GradientDescent(
    get_connection(),
    step=0.3,
    epoch_end_signal=save_epoch_weight
)
network_target_function = partial(target_function, bp_network)

plt.figure()
plt.title("Approximation function contour plot")
plt.xlabel("First weight")
plt.ylabel("Second weight")

draw_countour(
    np.linspace(-5, 5, 50),
    np.linspace(-5, 5, 50),
    network_target_function
)

cgnet_class = partial(algorithms.GradientDescent,
                      addons=[algorithms.LinearSearch],
                      search_method='golden')
momentum_class = partial(algorithms.Momentum, batch_size='full')

algorithms = (
    (algorithms.GradientDescent, 'Gradient Descent', 'k'),
    (momentum_class, 'Momentum', 'g'),
    (algorithms.RPROP, 'RPROP', 'm'),
    (algorithms.IRPROPPlus, 'iRPROP+', 'r'),
    (cgnet_class, 'Gradient Descent and\nGolden Search', 'y'),
)

patches = []
for algorithm, algorithm_name, color in algorithms:
    print("Train '{}' network".format(algorithm_name))
    quiver_patch = draw_quiver(algorithm, algorithm_name, color)
    patches.append(quiver_patch)

print("Plot training results")
plt.legend(handles=patches)
plt.show()
