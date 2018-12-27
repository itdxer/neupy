import matplotlib.pyplot as plt
from neupy import algorithms, utils

from helpers import plot_2d_grid, make_circle


plt.style.use('ggplot')
utils.reproducible()


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

    red, blue = ('#E24A33', '#348ABD')
    n_columns = len(configurations)

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

        plt.subplot(1, n_columns, index)

        plt.title(conf['title'])
        plt.scatter(*data.T, color=blue, alpha=0.05)
        plt.scatter(*sofm.weight, color=red)

        weights = sofm.weight.reshape((2, GRID_HEIGHT, GRID_WIDTH))
        plot_2d_grid(weights, color=red, hexagon=conf['use_hexagon_grid'])

    plt.show()
