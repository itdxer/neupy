from itertools import product

import matplotlib.pyplot as plt


def plot_2d_grid(weights, ax=None, color='green'):
    if weights.ndim != 3:
        raise ValueError("Number of dimensions should be equal to 3 "
                         "(shape: (2, height, width)), got {} instead"
                         "".format(weights.ndim))

    n_features, grid_height, grid_width = weights.shape

    if n_features != 2:
        raise ValueError("First dimension should be equal to 2")

    if ax is None:
        ax = plt.gca()

    actions = ((-1, 0), (0, -1), (1, 0), (0, 1))

    for neuron_x, neuron_y in product(range(grid_height), range(grid_width)):
        for shift_x, shift_y in actions:
            neigbour_x = neuron_x + shift_x
            neigbour_y = neuron_y + shift_y

            if 0 <= neigbour_x < grid_height and 0 <= neigbour_y < grid_width:
                neurons_x_coords = (neuron_x, neigbour_x)
                neurons_y_coords = (neuron_y, neigbour_y)

                neurons = weights[:, neurons_x_coords, neurons_y_coords]
                ax.plot(*neurons, color=color)
