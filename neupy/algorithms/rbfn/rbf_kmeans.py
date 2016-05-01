from __future__ import division

from numpy import (zeros, argmin, argwhere, take, sum as np_sum,
                   any as np_any, abs as np_abs)
from numpy.linalg import norm

from neupy.utils import format_data
from neupy.core.properties import IntProperty
from neupy.algorithms.gd import NoStepSelection
from neupy.network.base import BaseNetwork
from neupy.network.learning import UnsupervisedLearning


__all__ = ('RBFKMeans',)


class RBFKMeans(NoStepSelection, UnsupervisedLearning, BaseNetwork):
    """ Radial basis function K-means for clustering.

    Parameters
    ----------
    n_clusters : int
        number of clusters in dataset.
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Attributes
    ----------
    centers : numpy array [n_clusters, n_futures]
        After training this property will contain coordinates
        to cluster centers.

    Methods
    -------
    {UnsupervisedLearning.train}
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

    def __init__(self, **options):
        self.centers = None
        super(RBFKMeans, self).__init__(**options)

    def predict(self, input_data):
        input_data = format_data(input_data)

        centers = self.centers
        classes = zeros((input_data.shape[0], 1))

        for i, value in enumerate(input_data):
            classes[i] = argmin(norm(centers - value, axis=1))

        return classes

    def train_epoch(self, input_train, target_train):
        centers = self.centers
        old_centers = centers.copy()
        output_train = self.predict(input_train)

        for i, center in enumerate(centers):
            positions = argwhere(output_train[:, 0] == i)

            if not np_any(positions):
                continue

            class_data = take(input_train, positions, axis=0)
            centers[i, :] = (1 / len(class_data)) * np_sum(class_data, axis=0)

        return np_abs(old_centers - centers)

    def train(self, input_train, epsilon=1e-5, epochs=100):
        n_clusters = self.n_clusters
        input_train = format_data(input_train)

        if input_train.shape[0] <= n_clusters:
            raise ValueError("Count of clusters must be less than count of "
                             "input data.")

        self.centers = input_train[:n_clusters, :].copy()
        super(RBFKMeans, self).train(input_train, epsilon=epsilon,
                                     epochs=epochs)
