import numpy as np

from neupy.utils import format_data
from neupy.exceptions import NotTrained
from neupy.core.properties import IntProperty, ParameterProperty
from neupy.algorithms.base import BaseNetwork
from neupy import init


__all__ = ('Oja',)


class Oja(BaseNetwork):
    """
    Oja is an unsupervised technique used for the
    dimensionality reduction tasks.

    Notes
    -----
    - In practice use step as very small value.
      For instance, value ``1e-7`` can be a good choice.

    - Normalize the input data before use Oja algorithm.
      Input data shouldn't contains large values.

    - Set up smaller values for weight if error for a few
      first iterations is big compare to the input values scale.
      For instance, if your input data have values between
      ``0`` and ``1`` error value equal to ``100`` is big.

    - During the training network report mean absolute error (MAE)

    Parameters
    ----------
    minimized_data_size : int
        Expected number of features after minimization,
        defaults to ``1``.

    weight : array-like or ``None``
        Defines networks weights.
        Defaults to :class:`XavierNormal() <neupy.init.XavierNormal>`.

    {BaseNetwork.Parameters}

    Methods
    -------
    reconstruct(X)
        Reconstruct original dataset from the minimized input.

    train(X, epochs=100)
        Trains the network to the data X. Network trains until maximum
        number of ``epochs`` was reached.

    predict(X)
        Returns hidden representation of the input data ``X``. Basically,
        it applies dimensionality reduction.

    {BaseSkeleton.fit}

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
    >>> ojanet.train(data, epochs=100)
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
    minimized_data_size = IntProperty(minval=1)
    weight = ParameterProperty(default=init.XavierNormal())

    def one_training_update(self, X, y_train):
        weight = self.weight

        minimized = np.dot(X, weight)
        reconstruct = np.dot(minimized, weight.T)
        error = X - reconstruct

        weight += self.step * np.dot(error.T, minimized)
        mae = np.sum(np.abs(error)) / X.size

        # Clean objects from the memory
        del minimized
        del reconstruct
        del error

        return mae

    def train(self, X, epochs=100):
        X = format_data(X)
        n_input_features = X.shape[1]

        if isinstance(self.weight, init.Initializer):
            weight_shape = (n_input_features, self.minimized_data_size)
            self.weight = self.weight.sample(weight_shape, return_array=True)

        if n_input_features != self.weight.shape[0]:
            raise ValueError(
                "Invalid number of features. Expected {}, got {}"
                "".format(self.weight.shape[0], n_input_features))

        super(Oja, self).train(X, epochs=epochs)

    def reconstruct(self, X):
        if not isinstance(self.weight, np.ndarray):
            raise NotTrained("Network hasn't been trained yet")

        X = format_data(X)
        if X.shape[1] != self.minimized_data_size:
            raise ValueError(
                "Invalid input data feature space, expected "
                "{}, got {}.".format(X.shape[1], self.minimized_data_size))

        return np.dot(X, self.weight.T)

    def predict(self, X):
        if not isinstance(self.weight, np.ndarray):
            raise NotTrained("Network hasn't been trained yet")

        X = format_data(X)
        return np.dot(X, self.weight)
