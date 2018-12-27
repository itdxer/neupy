from itertools import product

import matplotlib.pyplot as plt
from neupy import algorithms, utils, init

from helpers import plot_2d_grid, make_circle, make_elipse, make_square


plt.style.use('ggplot')
utils.reproducible()


if __name__ == '__main__':
    GRID_WIDTH = 4
    GRID_HEIGHT = 4

    datasets = [
        make_square(),
        make_circle(),
        make_elipse(corr=0.7),
    ]
    configurations = [{
        'weight_init': init.Uniform(0, 1),
        'title': 'Random uniform initialization',
    }, {
        'weight_init': 'sample_from_data',
        'title': 'Sampled from the data',
    }, {
        'weight_init': 'init_pca',
        'title': 'Initialize with PCA',
    }]

    plt.figure(figsize=(15, 15))
    plt.title("Compare weight initialization methods for SOFM")

    red, blue = ('#E24A33', '#348ABD')
    n_columns = len(configurations)
    n_rows = len(datasets)
    index = 1

    for data, conf in product(datasets, configurations):
        sofm = algorithms.SOFM(
            n_inputs=2,
            features_grid=(GRID_HEIGHT, GRID_WIDTH),

            verbose=True,
            shuffle_data=True,
            weight=conf['weight_init'],

            learning_radius=8,
            reduce_radius_after=5,

            std=2,
            reduce_std_after=5,

            step=0.3,
            reduce_step_after=5,
        )

        if not sofm.initialized:
            sofm.init_weights(data)

        plt.subplot(n_rows, n_columns, index)

        plt.title(conf['title'])
        plt.scatter(*data.T, color=blue, alpha=0.05)
        plt.scatter(*sofm.weight, color=red)

        weights = sofm.weight.reshape((2, GRID_HEIGHT, GRID_WIDTH))
        plot_2d_grid(weights, color=red)

        index += 1

    plt.show()
