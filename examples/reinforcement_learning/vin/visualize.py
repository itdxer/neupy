from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from neupy import environment, storage

from loaddata import load_data
from train_vin import parser, create_VIN
from settings import environments
from evaluation import detect_trajectory


def plot_grid_and_trajectory(f_next_step, grid, coords):
    image_shape = grid[0, 0].shape
    trajectory = detect_trajectory(f_next_step, grid[0], coords)

    trajectory_grid = np.zeros(image_shape)
    trajectory_grid[trajectory[:, 0], trajectory[:, 1]] = 1

    start_position = np.zeros(image_shape)
    start_position[coords[0], coords[1]] = 1

    # Grid world map
    plt.imshow(grid[0, 0], interpolation='none', cmap='binary')

    # Trajectory
    cmap = plt.cm.Reds
    cmap.set_under(alpha=0)
    plt.imshow(trajectory_grid, interpolation='none',
               cmap=cmap, clim=[0.1, 1.6])

    # Start position
    cmap = plt.cm.Reds
    cmap.set_under(alpha=0)
    plt.imshow(start_position, interpolation='none',
               cmap=cmap, clim=[0.1, 1.1])

    # Goal position
    cmap = plt.cm.Greens
    cmap.set_under(alpha=0)
    plt.imshow(grid[0, 1] / 10., interpolation='none',
               cmap=cmap, clim=[0.1, 1.1])

    # Intercections between trajectories and obstacles
    cmap = plt.cm.Blues
    cmap.set_under(alpha=0)
    plt.imshow(np.bitwise_and(trajectory_grid == 1, grid[0, 0] == 1),
               interpolation='none', cmap=cmap, clim=[0.1, 1.1])


def sample_random_position(grid):
    obstacles_grid = grid[0, 0]
    x_coords, y_coords = np.argwhere(obstacles_grid == 0).T
    position = np.random.randint(x_coords.size)
    return (x_coords[position], y_coords[position])


if __name__ == '__main__':
    environment.speedup()

    args = parser.parse_args()
    env = environments[args.imsize]

    x_test, _, _, _ = load_data(env['test_data_file'])

    VIN = create_VIN(
        env['input_image_shape'],
        n_hidden_filters=150,
        n_state_filters=10,
        k=env['k'],
    )
    storage.load(VIN, env['pretrained_network_file'])
    predict = VIN.compile()

    plt.figure(figsize=(8, 8))
    gridspec = gridspec.GridSpec(5, 4, height_ratios=[0, 2, 2, 2, 2])
    gridspec.update(wspace=0.1, hspace=0.1)

    plt.suptitle('Trajectories between two points predicted by VIN ')

    plt.subplot(gridspec[0, :])
    plt.legend(
        handles=[
            mpatches.Patch(color='#A71C1B', label='Start'),
            mpatches.Patch(color='#F35D47', label='Trajectory'),
            mpatches.Patch(color='#007035', label='Goal'),
        ],
        loc=3,
        ncol=3,
        mode="expand",
        borderaxespad=0.,
        bbox_to_anchor=(0., 1.02, 1., .102),
    )
    plt.axis('off')

    for row, col in product(range(1, 5), range(4)):
        example_id = np.random.randint(x_test.shape[0])

        grid = np.expand_dims(x_test[example_id], axis=0)
        coords = sample_random_position(grid)

        plt.subplot(gridspec[row, col])
        plot_grid_and_trajectory(predict, grid, coords)
        plt.axis('off')

    plt.show()
