from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from neupy import algorithms, layers
from plots import draw_countour, weight_quiver


# Settings
TOTAL_EPOCHS = 30

# Initialize test data
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


def save_weight_in_epoch(net):
    """ Signal processor which save weight update for every
    epoch.
    """
    global weights
    input_layer_weight = net.train_layers[0].weight.copy()
    weights[:, net.epoch + 1:net.epoch + 2] = input_layer_weight


def get_connection():
    """ Generate new connections every time when we call it """
    input_layer = layers.SigmoidLayer(2, weight=default_weight.copy())
    output_layer = layers.OutputLayer(1)
    return input_layer > output_layer


# Setup default networks settings which we will use in all algorithms
# which we will vizualize.
network_settings = dict(
    # Network
    step=0.3,
    use_bias=False,
    # Signals
    train_epoch_end_signal=save_weight_in_epoch,
    verbose=False,
)


def draw_quiver(network_class, name, color='r'):
    """ Train algorithm and draw quiver for every epoch update
    for this algorithm.
    """
    global weights

    bpn = network_class(get_connection(), **network_settings)
    # 1000 is an upper limit for all network epochs, later we
    # will fix it size
    weights = np.zeros((2, 1000))
    weights[:, 0:1] = default_weight.copy()
    bpn.train(input_data, target_data, epsilon=0.125)
    weights = weights[:, :bpn.epoch + 1]
    weight_quiver(weights, color=color)

    label = "{name} ({n} steps)".format(name=name, n=bpn.epoch)
    return mpatches.Patch(color=color, label=label)


def target_function(network, x, y):
    network.input_layer.weight = np.array([[x], [y]])
    predicted = network.predict(input_data)
    return network.error(predicted, target_data)


# Get data for countour plot
bp_network = algorithms.Backpropagation(get_connection(), **network_settings)
network_target_function = partial(target_function, bp_network)

plt.figure()
plt.title("Approximation function contour plot")
plt.xlabel("First weight")
plt.ylabel("Second weight")

draw_countour(
    np.linspace(-4.5, 4, 50),
    np.linspace(-4.5, 4, 50),
    network_target_function
)

cgnet_class = partial(algorithms.ConjugateGradient,
                      optimizations=[algorithms.LinearSearch])

algorithms = (
    (algorithms.Backpropagation, 'Gradient Descent', 'k'),
    (algorithms.Momentum, 'Momentum', 'm'),
    (algorithms.RPROP, 'RPROP', 'c'),
    (cgnet_class, 'Conjugate Gradient', 'y'),
)

patches = []
for network_params in algorithms:
    quiver_patch = draw_quiver(*network_params)
    patches.append(quiver_patch)

plt.legend(handles=patches)
plt.show()
