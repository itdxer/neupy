import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from neupy import algorithms, utils


plt.style.use('ggplot')
utils.reproducible()


if __name__ == '__main__':
    ggplot_colors = plt.rcParams['axes.prop_cycle']
    colors = np.array([c['color'] for c in ggplot_colors])

    dataset = datasets.load_iris()
    # use only two features in order
    # to make visualization simpler
    data = dataset.data[:, [2, 3]]
    target = dataset.target

    sofm = algorithms.SOFM(
        # Use only two features for the input
        n_inputs=2,

        # Number of outputs defines number of features
        # in the SOFM or in terms of clustering - number
        # of clusters
        n_outputs=20,

        # In clustering application we will prefer that
        # clusters will be updated independently from each
        # other. For this reason we set up learning radius
        # equal to zero
        learning_radius=0,

        # Training step size or learning rate
        step=0.25,

        # Shuffles dataset before every training epoch.
        shuffle_data=True,

        # Instead of generating random weights
        # (features / cluster centers) SOFM will sample
        # them from the data. Which means that after
        # initialization step 3 random data samples will
        # become cluster centers
        weight='sample_from_data',

        # Shows training progress in terminal
        verbose=True,
    )
    sofm.train(data, epochs=200)

    plt.title('Clustering iris dataset with SOFM')
    plt.xlabel('Feature #3')
    plt.ylabel('Feature #4')

    plt.scatter(*data.T, c=colors[target], s=100, alpha=1)
    cluster_centers = plt.scatter(*sofm.weight, s=300, c=colors[3])

    plt.legend([cluster_centers], ['Cluster center'], loc='upper left')
    plt.show()
