from __future__ import division

import numpy as np
from tqdm import tqdm
from neupy.utils import asfloat


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


def invalid_coordinate(grid, position):
    height, width = grid.shape
    x_coord, y_coord = position
    return not (0 <= x_coord < height) or not (0 <= y_coord < width)


def shortest_path_length(grid, start, finish):
    if start == finish:
        return 0

    path_grid = np.zeros(grid.shape, dtype=int)
    path_grid[start] = 1
    positions = [start]

    while positions:
        current = positions.pop(0)
        current_value = path_grid[current]

        for action in actions:
            next_coord = action(*current)

            if invalid_coordinate(grid, next_coord):
                continue

            not_obstacle = grid[next_coord] != 1
            not_observed = path_grid[next_coord] == 0

            if not_obstacle and not_observed:
                if next_coord == finish:
                    return current_value

                path_grid[next_coord] = current_value + 1
                positions.append(next_coord)

    # Impossible to generate path between two points
    return None


def detect_trajectory(f_next_step, grid, coords, max_iter=200):
    target_coords = get_target_coords(grid)
    input_grid = asfloat(np.expand_dims(grid, axis=0))
    grid = grid[0]

    trajectory = [coords]
    coord_x, coord_y = coords

    for i in range(max_iter):
        current_position = (coord_x, coord_y)

        if target_coords == current_position:
            break

        if grid[current_position] == 1:
            # Interrupt trajectory detection procedure, because
            # current position is located on the obstacle.
            return None

        step = f_next_step(input_grid,
                           int_as_2d_array(coord_x),
                           int_as_2d_array(coord_y))
        step = np.argmax(step, axis=1)
        action = actions[step[0]]

        coord_x, coord_y = action(coord_x, coord_y)
        trajectory.append((coord_x, coord_y))

    return np.array(trajectory)


def int_as_2d_array(number):
    return asfloat(np.array([[number]]))


def get_target_coords(gridworld_image):
    goal_channel = gridworld_image[1]
    return np.unravel_index(goal_channel.argmax(), goal_channel.shape)


def evaluate_accuracy(predict, x_test, s1_test, s2_test):
    n_grids = x_test.shape[0]
    state_batch_size = s1_test.shape[1]

    total = n_grids * state_batch_size
    correct = 0
    differences = 0

    data = tqdm(zip(x_test, s1_test, s2_test),
                total=n_grids, desc='Evaluating accuracy')

    for grid, x_coords, y_coords in data:
        finish = get_target_coords(grid)

        for start in zip(x_coords, y_coords):
            len_shortest = shortest_path_length(grid[0], start, finish)

            max_iter = 2 * len_shortest
            trajectory = detect_trajectory(predict, grid, start, max_iter)

            if trajectory is not None:
                len_predicted = len(trajectory) - 1
            else:
                len_predicted = max_iter

            differences += len_predicted - len_shortest

            if len_predicted == len_shortest:
                correct += 1

    accuracy = correct / total
    loss = differences / total

    print("Success rate: {:.2%}".format(accuracy))
    print("Prediction loss: {:.4f}".format(loss))
