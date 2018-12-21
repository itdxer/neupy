from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import datasets
from neupy import algorithms, utils


utils.reproducible()

GRID_HEIGHT = 40
GRID_WIDTH = 40

digits = datasets.load_digits()
data = digits.data

sofm = algorithms.SOFM(
    n_inputs=64,
    features_grid=(GRID_HEIGHT, GRID_WIDTH),

    # Learning radius defines area within which we find
    # winning neuron neighbours. The higher the value
    # the more values we will be updated after each iteration.
    learning_radius=5,
    # Every 20 epochs learning radius will be reduced by 1.
    reduce_radius_after=20,

    step=0.5,
    std=1,

    shuffle_data=True,
    verbose=True,
)

sofm.train(data, epochs=100)
clusters = sofm.predict(data).argmax(axis=1)

print("Building visualization...")
plt.figure(figsize=(12, 11))

grid = gridspec.GridSpec(GRID_HEIGHT, GRID_WIDTH)
grid.update(wspace=0, hspace=0)

for row_id in range(GRID_HEIGHT):
    print("Progress: {:.2%}".format(row_id / GRID_HEIGHT))

    for col_id in range(GRID_WIDTH):
        index = row_id * GRID_HEIGHT + col_id
        clustered_samples = data[clusters == index]

        if len(clustered_samples) > 0:
            # We take the first sample, but it can be any
            # sample from this cluster
            sample = clustered_samples[0]

        else:
            # If we don't have samples in cluster then
            # it means that there is a gap in space
            sample = np.zeros(64)

        plt.subplot(grid[index])
        plt.imshow(sample.reshape((8, 8)), cmap='Greys')
        plt.axis('off')

print("Visualization has been built succesfully")
plt.show()
