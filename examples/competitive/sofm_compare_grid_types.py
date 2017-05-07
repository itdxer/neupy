import numpy as np
import matplotlib.pyplot as plt
from neupy import algorithms, environment

from utils import plot_2d_grid


plt.style.use('ggplot')
environment.reproducible()


def make_circle():
    data = np.random.random((10000, 2))
    x, y = data[:, 0], data[:, 1]

    distance_from_center = ((x - 0.5) ** 2 + (y - 0.5) ** 2)
    return data[distance_from_center <= 0.5 ** 2]


if __name__ == '__main__':
    GRID_WIDTH = 10
    GRID_HEIGHT = 10

    configurations = [{
        'grid_type': 'hexagon',
        'use_hexagon_grid': True,
        'title': 'Using hexagon grid',
    }, {
        'grid_type': 'rect',
        'use_hexagon_grid': False,
        'title': 'Using regcangular grid',
    }]

    data = make_circle()
    plt.figure(figsize=(12, 5))

    for index, conf in enumerate(configurations, start=1):
        sofm = algorithms.SOFM(
            n_inputs=2,
            features_grid=(GRID_HEIGHT, GRID_WIDTH),

            verbose=True,
            shuffle_data=True,
            grid_type=conf['grid_type'],

            learning_radius=8,
            reduce_radius_after=5,

            std=2,
            reduce_std_after=5,

            step=0.3,
            reduce_step_after=5,
        )
        sofm.train(data, epochs=40)

        red, blue = ('#E24A33', '#348ABD')
        n_columns = len(configurations)

        plt.subplot(1, n_columns, index)

        plt.title(conf['title'])
        plt.scatter(*data.T, color=blue, alpha=0.05)
        plt.scatter(*sofm.weight, color=red)

        weights = sofm.weight.reshape((2, GRID_HEIGHT, GRID_WIDTH))
        plot_2d_grid(weights, color=red, hexagon=conf['use_hexagon_grid'])

    plt.show()
