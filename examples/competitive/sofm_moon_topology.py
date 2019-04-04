import matplotlib.pyplot as plt
from sklearn import datasets
from neupy import algorithms, utils

from helpers import plot_2d_grid


plt.style.use('ggplot')
utils.reproducible()


if __name__ == '__main__':
    GRID_WIDTH = 20
    GRID_HEIGHT = 1

    data, targets = datasets.make_moons(n_samples=400, noise=0.1)
    data = data[targets == 1]

    sofm = algorithms.SOFM(
        n_inputs=2,
        features_grid=(GRID_HEIGHT, GRID_WIDTH),

        verbose=True,
        shuffle_data=True,

        # The winning neuron will be selected based on the
        # Euclidean distance. For this task it's important
        # that distance is Euclidean. Other distances will
        # not give us the same results.
        distance='euclid',

        learning_radius=2,
        # Reduce learning radius by 1 after every 20 epochs.
        # Learning radius will be equal to 2 during first
        # 20 epochs and on the 21st epoch it will be equal to 1.
        reduce_radius_after=20,

        # 2 Means that neighbour neurons will have high learning
        # rates during the first iterations
        std=2,
        # Defines a rate at which parameter `std` will be reduced.
        # Reduction is monotonic and reduces after each epoch.
        # In 50 epochs std = 2 / 2 = 1 and after 100 epochs
        # std = 2 / 3 and so on.
        reduce_std_after=50,

        # Step (or learning rate)
        step=0.3,
        # Defines a rate at which parameter `step` will reduced.
        # Reduction is monotonic and reduces after each epoch.
        # In 50 epochs step = 0.3 / 2 = 0.15 and after 100 epochs
        # std = 0.3 / 3 = 0.1 and so on.
        reduce_step_after=50,
    )
    sofm.train(data, epochs=20)

    red, blue = ('#E24A33', '#348ABD')

    plt.scatter(*data.T, color=blue)
    plt.scatter(*sofm.weight, color=red)

    weights = sofm.weight.reshape((2, GRID_HEIGHT, GRID_WIDTH))
    plot_2d_grid(weights, color=red)

    plt.show()
