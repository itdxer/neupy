import numpy as np

from neupy.utils import format_data, NotTrainedException
from neupy.core.properties import IntProperty, ParameterProperty
from neupy.network.base import BaseNetwork
from neupy.network.learning import UnsupervisedLearningMixin
from neupy import init


__all__ = ('Oja',)


class Oja(UnsupervisedLearningMixin, BaseNetwork):
    """
    Oja unsupervised algorithm that minimize input data feature
    space.

    Notes
    -----
    * In practice use step as very small value. For example ``1e-7``.
    * Normalize the input data before use Oja algorithm. Input data \
    shouldn't contains large values.
    * Set up smaller values for weights if error for a few first iterations \
    is big compare to the input values scale. For example, if your input \
    data have values between 0 and 1 error value equal to 100 is big.

    Parameters
    ----------
    minimized_data_size : int
        Expected number of features after minimization, defaults to ``1``
    weights : array-like or ``None``
        Predefine default weights which controll your data in two sides.
        If weights are, ``None`` before train algorithms generate random
        weights. Defaults to ``None``.
    {BaseNetwork.step}
    {BaseNetwork.show_epoch}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Methods
    -------
    reconstruct(input_data):
        Reconstruct your minimized data.
    {BaseSkeleton.predict}
    {UnsupervisedLearningMixin.train}
    {BaseSkeleton.fit}

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
    minimized_data_size = IntProperty(minval=1)
    weights = ParameterProperty(default=init.XavierNormal())

    def init_properties(self):
        del self.shuffle_data
        super(Oja, self).init_properties()

    def train_epoch(self, input_data, target_train):
        weights = self.weights

        minimized = np.dot(input_data, weights)
        reconstruct = np.dot(minimized, weights.T)
        error = input_data - reconstruct

        weights += self.step * np.dot(error.T, minimized)

        mae = np.sum(np.abs(error)) / input_data.size

        # Clear memory
        del minimized
        del reconstruct
        del error

        return mae

    def train(self, input_data, epsilon=1e-2, epochs=100):
        input_data = format_data(input_data)
        n_input_features = input_data.shape[1]

        if isinstance(self.weights, init.Initializer):
            weight_shape = (n_input_features, self.minimized_data_size)
            self.weights = self.weights.sample(weight_shape)

        if n_input_features != self.weights.shape[0]:
            raise ValueError(
                "Invalid number of features. Expected {}, got {}".format(
                    self.weights.shape[0],
                    n_input_features
                )
            )

        super(Oja, self).train(input_data, epsilon=epsilon, epochs=epochs)

    def reconstruct(self, input_data):
        if not isinstance(self.weights, np.ndarray):
            raise NotTrainedException("Train network before use "
                                      "reconstruct method.")

        input_data = format_data(input_data)
        if input_data.shape[1] != self.minimized_data_size:
            raise ValueError(
                "Invalid input data feature space, expected "
                "{}, got {}.".format(
                    input_data.shape[1],
                    self.minimized_data_size
                )
            )

        return np.dot(input_data, self.weights.T)

    def predict(self, input_data):
        if not isinstance(self.weights, np.ndarray):
            raise NotTrainedException("Train network before use "
                                      "prediction method.")

        input_data = format_data(input_data)
        return np.dot(input_data, self.weights)
