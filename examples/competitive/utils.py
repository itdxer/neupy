from itertools import product

import numpy as np
import matplotlib.pyplot as plt


def iter_neighbours(weights, hexagon=False):
    _, grid_height, grid_width = weights.shape

    hexagon_even_actions = ((-1, 0), (0, -1), (1, 0), (0, 1), (1, 1), (-1, 1))
    hexagon_odd_actions = ((-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1))
    rectangle_actions = ((-1, 0), (0, -1), (1, 0), (0, 1))

    for neuron_x, neuron_y in product(range(grid_height), range(grid_width)):
        neighbours = []

        if hexagon and neuron_x % 2 == 1:
            actions = hexagon_even_actions
        elif hexagon:
            actions = hexagon_odd_actions
        else:
            actions = rectangle_actions

        for shift_x, shift_y in actions:
            neigbour_x = neuron_x + shift_x
            neigbour_y = neuron_y + shift_y

            if 0 <= neigbour_x < grid_height and 0 <= neigbour_y < grid_width:
                neighbours.append((neigbour_x, neigbour_y))

        yield (neuron_x, neuron_y), neighbours


def plot_2d_grid(weights, ax=None, color='green', hexagon=False):
    if weights.ndim != 3:
        raise ValueError("Number of dimensions should be equal to 3 "
                         "(shape: (2, height, width)), got {} instead"
                         "".format(weights.ndim))

    n_features = weights.shape[0]

    if n_features != 2:
        raise ValueError("First dimension should be equal to 2")

    if ax is None:
        ax = plt.gca()

    for (neuron_x, neuron_y), neighbours in iter_neighbours(weights, hexagon):
        for (neigbour_x, neigbour_y) in neighbours:
            neurons_x_coords = (neuron_x, neigbour_x)
            neurons_y_coords = (neuron_y, neigbour_y)

            neurons = weights[:, neurons_x_coords, neurons_y_coords]
            ax.plot(*neurons, color=color)


def make_square():
    return np.random.random((10000, 2))


def make_circle():
    data = make_square()
    x, y = data[:, 0], data[:, 1]

    distance_from_center = ((x - 0.5) ** 2 + (y - 0.5) ** 2)
    return data[distance_from_center <= 0.5 ** 2]


def make_elipse(corr=0.8):
    projection = np.array([
        [corr, 1 - corr],
        [1 - corr, corr]])

    data = make_circle()
    return data.dot(projection)
