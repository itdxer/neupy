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

    Parameters
    ----------
    minimized_data_size : int
        Expected number of features after minimization,
        defaults to ``1``.

    weight : array-like or ``None``
        Defines networks weights.
        Defaults to :class:`XavierNormal() <neupy.init.XavierNormal>`.

    {BaseNetwork.step}

    {BaseNetwork.show_epoch}

    {BaseNetwork.epoch_end_signal}

    {BaseNetwork.train_end_signal}

    {Verbose.verbose}

    Methods
    -------
    reconstruct(input_data)
        Reconstruct original dataset from the minimized input.

    train(input_data, epsilon=1e-2, epochs=100)
        Trains algorithm based on the input dataset.
        For the dimensionality reduction input dataset
        assumes to be also a target.

    {BaseSkeleton.predict}

    {BaseSkeleton.fit}

    Raises
    ------
    ValueError
        - Triggers when you try to reconstruct output
          without training.

        - Invalid number of input data features for the
          ``train`` and ``reconstruct`` methods.

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
    minimized_data_size = IntProperty(minval=1)
    weight = ParameterProperty(default=init.XavierNormal())

    def train_epoch(self, input_data, target_train):
        weight = self.weight

        minimized = np.dot(input_data, weight)
        reconstruct = np.dot(minimized, weight.T)
        error = input_data - reconstruct

        weight += self.step * np.dot(error.T, minimized)

        mae = np.sum(np.abs(error)) / input_data.size

        # Clean objects from the memory
        del minimized
        del reconstruct
        del error

        return mae

    def train(self, input_data, epsilon=1e-2, epochs=100):
        input_data = format_data(input_data)
        n_input_features = input_data.shape[1]

        if isinstance(self.weight, init.Initializer):
            weight_shape = (n_input_features, self.minimized_data_size)
            self.weight = self.weight.sample(weight_shape)

        if n_input_features != self.weight.shape[0]:
            raise ValueError(
                "Invalid number of features. Expected {}, got {}".format(
                    self.weight.shape[0],
                    n_input_features
                )
            )

        super(Oja, self).train(input_data, epsilon=epsilon, epochs=epochs)

    def reconstruct(self, input_data):
        if not isinstance(self.weight, np.ndarray):
            raise NotTrained("Network hasn't been trained yet")

        input_data = format_data(input_data)
        if input_data.shape[1] != self.minimized_data_size:
            raise ValueError(
                "Invalid input data feature space, expected "
                "{}, got {}.".format(
                    input_data.shape[1],
                    self.minimized_data_size
                )
            )

        return np.dot(input_data, self.weight.T)

    def predict(self, input_data):
        if not isinstance(self.weight, np.ndarray):
            raise NotTrained("Network hasn't been trained yet")

        input_data = format_data(input_data)
        return np.dot(input_data, self.weight)
