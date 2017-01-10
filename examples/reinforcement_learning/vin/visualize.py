from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from neupy.utils import asfloat
from neupy import environment, storage

from loaddata import CURRENT_DIR, TRAIN_DATA, TEST_DATA, load_data
from vin import PRETRAINED_NETWORK, create_VIN


actions = [
    lambda x, y: (x - 1, y),  # north
    lambda x, y: (x + 1, y),  # south
    lambda x, y: (x, y + 1),  # east
    lambda x, y: (x, y - 1),  # west
    lambda x, y: (x - 1, y + 1),  # north-east
    lambda x, y: (x - 1, y - 1),  # north-west
    lambda x, y: (x + 1, y + 1),  # south-east
    lambda x, y: (x + 1, y - 1),  # south-west
]


def get_target_coords(gridworld_image):
    goal_channel = gridworld_image[1]
    return np.unravel_index(goal_channel.argmax(), goal_channel.shape)


def int_as_2d_array(number):
    return asfloat(np.array([[number]]))


def detect_trajectory(f_next_step, grid, coords, max_iter=20):
    trajectory = [coords]
    coord_x, coord_y = coords
    target_coords = get_target_coords(grid[0])

    for i in range(max_iter):
        if target_coords == (coord_x, coord_y):
            break

        step = f_next_step(asfloat(grid),
                           int_as_2d_array(coord_x),
                           int_as_2d_array(coord_y))
        step = np.argmax(step, axis=1)
        action = actions[step[0]]

        coord_x, coord_y = action(coord_x, coord_y)
        trajectory.append((coord_x, coord_y))

    return np.array(trajectory)


def plot_grid_and_trajectory(f_next_step, grid, coords):
    image_shape = grid[0, 0].shape
    trajectory = detect_trajectory(f_next_step, grid, coords)

    trajectory_grid = np.zeros(image_shape)
    trajectory_grid[trajectory[:, 0], trajectory[:, 1]] = 1

    start_position = np.zeros(image_shape)
    start_position[coords[0], coords[1]] = 1

    # Grid world map
    plt.imshow(grid[0, 0], interpolation='nearest', cmap='binary')

    # Trajectory
    cmap = plt.cm.jet
    cmap.set_under(alpha=0)
    plt.imshow(trajectory_grid, interpolation='none',
               cmap=cmap, clim=[0.1, 1.1])

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

def sample_random_position(grid):
    obstacles_grid = grid[0, 0]
    x_coords, y_coords = np.argwhere(obstacles_grid == 0).T
    position = np.random.randint(x_coords.size)
    return (x_coords[position], y_coords[position])


if __name__ == '__main__':
    environment.speedup()
    x_test, _, _, _ = load_data(TRAIN_DATA)

    VIN = create_VIN(
        input_image_shape=(2, 8, 8),
        n_hidden_filters=150,
        n_state_filters=10,
        k=10,
    )
    storage.load(VIN, PRETRAINED_NETWORK)
    predict = VIN.compile()

    plt.figure(figsize=(8, 8))
    gridspec = gridspec.GridSpec(5, 4, height_ratios=[0, 2, 2, 2, 2])
    gridspec.update(wspace=0.1, hspace=0.1)

    plt.suptitle('Predicted by VIN trajectories between two points')

    plt.subplot(gridspec[0, :])
    plt.legend(
        handles=[
            mpatches.Patch(color='#A71C1B', label='Start'),
            mpatches.Patch(color='#F32919', label='Trajectory'),
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
