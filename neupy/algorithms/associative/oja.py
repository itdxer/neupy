from numpy import dot, abs as np_abs
from numpy.random import randn

from neupy.utils import format_data
from neupy.core.properties import NonNegativeIntProperty, ArrayProperty
from neupy.network.base import BaseNetwork
from neupy.network.learning import UnsupervisedLearning
from neupy.network.connections import FAKE_CONNECTION


__all__ = ('Oja',)


class Oja(UnsupervisedLearning, BaseNetwork):
    """ Oja unsupervised algorithm which minimize feature space.

    Notes
    -----
    * In practice use step as very small value. For example ``1e-7``.

    Parameters
    ----------
    minimized_data_size : int
        Expected number of features after minimization, defaults to ``1``
    weights : array-like or ``None``
        Predefine default weights which controll your data in two sides.
        If weights are, ``None`` before train algorithms generate random
        weights. Defaults to ``None``.
    {step}
    {show_epoch}
    {verbose}
    {full_signals}

    Methods
    -------
    reconstruct(input_data):
        Reconstruct your minimized data.
    {unsupervised_train_epsilon}
    {full_methods}

    Raises
    ------
    ValueError
        * Try reconstruct without training.
        * Invalid number of input data features for ``train`` and \
        ``reconstruct`` methods.

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> data = np.array([[2, 2], [1, 1], [4, 4], [5, 5]])
    >>>
    >>> ojanet = algorithms.Oja(
    ...     minimized_data_size=1,
    ...     step=0.01,
    ...     verbose=False
    ... )
    >>>
    >>> ojanet.train(data, epsilon=1e-5)
    >>> minimized = ojanet.predict(data)
    >>> minimized
    array([[-2.82843122],
           [-1.41421561],
           [-5.65686243],
           [-7.07107804]])
    >>> ojanet.reconstruct(minimized)
    array([[ 2.00000046,  2.00000046],
           [ 1.00000023,  1.00000023],
           [ 4.00000093,  4.00000093],
           [ 5.00000116,  5.00000116]])
    """
    minimized_data_size = NonNegativeIntProperty(min_size=1)
    weights = ArrayProperty()

    def __init__(self, **options):
        super(Oja, self).__init__(FAKE_CONNECTION, **options)

    def setup_defaults(self):
        del self.use_bias
        del self.error
        del self.shuffle_data
        super(Oja, self).setup_defaults()

    def train_epoch(self, input_data, target_train):
        weights = self.weights

        minimized = dot(input_data, weights)
        reconstruct = dot(minimized, weights.T)
        error = input_data - reconstruct

        weights += self.step * dot(error.T, minimized)

        return np_abs(error) / (input_data.shape[0] * input_data.shape[1])

    def train(self, input_data, epsilon=1e-5):
        input_data = format_data(input_data)
        n_input_features = input_data.shape[1]

        if self.weights is None:
            self.weights = randn(n_input_features, self.minimized_data_size)

        if n_input_features != self.weights.shape[0]:
            raise ValueError(
                "Invalid number of features. Expected {}, got {}".format(
                    self.weights.shape[0], n_input_features
                )
            )

        super(Oja, self).train(input_data, epsilon=epsilon)

    def reconstruct(self, input_data):
        if self.weights is None:
            raise ValueError("Train network before use reconstruct method.")

        input_data = format_data(input_data)
        if input_data.shape[1] != self.minimized_data_size:
            raise ValueError(
                "Invalid input data feature space, expected "
                "{}, got {}.".format(
                    input_data.shape[1], self.minimized_data_size
                )
            )

        return dot(input_data, self.weights.T)

    def predict(self, input_data):
        if self.weights is None:
            raise ValueError("Train network before use prediction method.")

        input_data = format_data(input_data)
        return dot(input_data, self.weights)
