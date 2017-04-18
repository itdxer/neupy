import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing
from neupy import algorithms, environment

from utils import iter_neighbours


plt.style.use('ggplot')
environment.reproducible()

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


if __name__ == '__main__':
    GRID_HEIGHT = 15
    GRID_WIDTH = 15

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

    plt.figure(figsize=(10, 10))

    for actual_class, cluster_index in zip(target, clusters):
        cluster_coords = divmod(cluster_index, GRID_HEIGHT)
        plt.plot(*cluster_coords, **class_parameters[actual_class])

    weight = sofm.weight.reshape((sofm.n_inputs, GRID_HEIGHT, GRID_WIDTH))
    heatmap = compute_heatmap(weight)
    plt.imshow(heatmap, cmap='Greys_r', interpolation='nearest')

    plt.axis('off')
    plt.colorbar()
    plt.show()
