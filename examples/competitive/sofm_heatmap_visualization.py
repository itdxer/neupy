import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing
from neupy import algorithms, utils

from utils import iter_neighbours


plt.style.use('ggplot')
utils.reproducible()

parser = argparse.ArgumentParser()
parser.add_argument('--expanded-heatmap', action='store_true')

class_parameters = [
    dict(
        marker='o',
        markeredgecolor='#E24A33',
        markersize=11,
        markeredgewidth=2,
        markerfacecolor='None',
    ),
    dict(
        marker='s',
        markeredgecolor='#348ABD',
        markersize=14,
        markeredgewidth=2,
        markerfacecolor='None',
    ),
]


def load_data():
    data, target = datasets.load_breast_cancer(return_X_y=True)

    scaler = preprocessing.MinMaxScaler()
    data = scaler.fit_transform(data)

    return data, target


def compute_heatmap(weight):
    heatmap = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    for (neuron_x, neuron_y), neighbours in iter_neighbours(weight):
        total_distance = 0

        for (neigbour_x, neigbour_y) in neighbours:
            neuron_vec = weight[:, neuron_x, neuron_y]
            neigbour_vec = weight[:, neigbour_x, neigbour_y]

            distance = np.linalg.norm(neuron_vec - neigbour_vec)
            total_distance += distance

        avg_distance = total_distance / len(neighbours)
        heatmap[neuron_x, neuron_y] = avg_distance

    return heatmap


def compute_heatmap_expanded(weight):
    heatmap = np.zeros((2 * GRID_HEIGHT - 1, 2 * GRID_WIDTH - 1))
    for (neuron_x, neuron_y), neighbours in iter_neighbours(weight):
        for (neigbour_x, neigbour_y) in neighbours:
            neuron_vec = weight[:, neuron_x, neuron_y]
            neigbour_vec = weight[:, neigbour_x, neigbour_y]

            distance = np.linalg.norm(neuron_vec - neigbour_vec)

            if neuron_x == neigbour_x and (neigbour_y - neuron_y) == 1:
                heatmap[2 * neuron_x, 2 * neuron_y + 1] = distance

            elif (neigbour_x - neuron_x) == 1 and neigbour_y == neuron_y:
                heatmap[2 * neuron_x + 1, 2 * neuron_y] = distance

    return heatmap


if __name__ == '__main__':
    args = parser.parse_args()

    GRID_HEIGHT = 20
    GRID_WIDTH = 20

    sofm = algorithms.SOFM(
        n_inputs=30,
        features_grid=(GRID_HEIGHT, GRID_WIDTH),

        learning_radius=4,
        reduce_radius_after=50,

        step=0.5,
        std=1,

        shuffle_data=True,
        verbose=True,
    )

    data, target = load_data()
    sofm.train(data, epochs=300)
    clusters = sofm.predict(data).argmax(axis=1)

    plt.figure(figsize=(13, 13))
    plt.title("Embedded 30-dimensional dataset using SOFM")

    for actual_class, cluster_index in zip(target, clusters):
        cluster_x, cluster_y = divmod(cluster_index, GRID_HEIGHT)
        parameters = class_parameters[actual_class]

        if args.expanded_heatmap:
            plt.plot(2 * cluster_x, 2 * cluster_y, **parameters)
        else:
            plt.plot(cluster_x, cluster_y, **parameters)

    weight = sofm.weight.reshape((sofm.n_inputs, GRID_HEIGHT, GRID_WIDTH))

    if args.expanded_heatmap:
        heatmap = compute_heatmap_expanded(weight)
    else:
        heatmap = compute_heatmap(weight)

    plt.imshow(heatmap, cmap='Greys_r', interpolation='nearest')

    plt.axis('off')
    plt.colorbar()
    plt.show()
