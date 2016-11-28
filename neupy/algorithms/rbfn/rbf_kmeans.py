from __future__ import division

import numpy as np
from numpy.linalg import norm

from neupy.utils import format_data
from neupy.core.properties import IntProperty, WithdrawProperty
from neupy.algorithms.gd import StepSelectionBuiltIn
from neupy.algorithms.base import BaseNetwork


__all__ = ('RBFKMeans',)


class RBFKMeans(StepSelectionBuiltIn, BaseNetwork):
    """
    Radial basis function K-means for clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.epoch_end_signal}

    {BaseNetwork.train_end_signal}

    {Verbose.verbose}

    Attributes
    ----------
    centers : array-like with shape (n_clusters, n_futures)
        Cluster centers.

    Methods
    -------
    train(input_train, epsilon=1e-5, epochs=100)
        Trains network.

    {BaseSkeleton.predict}

    {BaseSkeleton.fit}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy.algorithms import RBFKMeans
    >>>
    >>> data = np.array([
    ...     [0.11, 0.20],
    ...     [0.25, 0.32],
    ...     [0.64, 0.60],
    ...     [0.12, 0.42],
    ...     [0.70, 0.73],
    ...     [0.30, 0.27],
    ...     [0.43, 0.81],
    ...     [0.44, 0.87],
    ...     [0.12, 0.92],
    ...     [0.56, 0.67],
    ...     [0.36, 0.35],
    ... ])
    >>> rbfk_net = RBFKMeans(n_clusters=2, verbose=False)
    >>> rbfk_net.train(data, epsilon=1e-5)
    >>> rbfk_net.centers
    array([[ 0.228     ,  0.312     ],
           [ 0.48166667,  0.76666667]])
    >>>
    >>> new_data = np.array([[0.1, 0.1], [0.9, 0.9]])
    >>> rbfk_net.predict(new_data)
    array([[ 0.],
           [ 1.]])
    """
    n_clusters = IntProperty(minval=2)
    step = WithdrawProperty()

    def __init__(self, **options):
        self.centers = None
        super(RBFKMeans, self).__init__(**options)

    def predict(self, input_data):
        input_data = format_data(input_data)

        centers = self.centers
        classes = np.zeros((input_data.shape[0], 1))

        for i, value in enumerate(input_data):
            classes[i] = np.argmin(norm(centers - value, axis=1))

        return classes

    def train_epoch(self, input_train, target_train):
        centers = self.centers
        old_centers = centers.copy()
        output_train = self.predict(input_train)

        for i, center in enumerate(centers):
            positions = np.argwhere(output_train[:, 0] == i)

            if not np.any(positions):
                continue

            class_data = np.take(input_train, positions, axis=0)
            centers[i, :] = (1 / len(class_data)) * np.sum(class_data, axis=0)

        return np.abs(old_centers - centers)

    def train(self, input_train, epsilon=1e-5, epochs=100):
        n_clusters = self.n_clusters
        input_train = format_data(input_train)
        n_samples = input_train.shape[0]

        if n_samples <= n_clusters:
            raise ValueError("Number of samples in the dataset is less than "
                             "spcified number of clusters. Got {} samples, "
                             "expected at least {} (for {} clusters)"
                             "".format(n_samples, n_clusters + 1, n_clusters))

        self.centers = input_train[:n_clusters, :].copy()
        super(RBFKMeans, self).train(input_train, epsilon=epsilon,
                                     epochs=epochs)
