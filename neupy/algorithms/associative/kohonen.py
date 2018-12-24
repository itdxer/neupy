import numpy as np

from neupy.utils import format_data
from .base import BaseAssociative


__all__ = ('Kohonen',)


class Kohonen(BaseAssociative):
    """
    Kohonen Neural Network used for unsupervised learning.

    Parameters
    ----------
    {BaseAssociative.Parameters}

    Methods
    -------
    {BaseAssociative.Methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms, layers, utils
    >>>
    >>> utils.reproducible()
    >>>
    >>> X = np.array([
    ...     [0.1961,  0.9806],
    ...     [-0.1961,  0.9806],
    ...     [0.9806,  0.1961],
    ...     [0.9806, -0.1961],
    ...     [-0.5812, -0.8137],
    ...     [-0.8137, -0.5812],
    ... ])
    >>>
    >>> kohonet = algorithms.Kohonen(
    ...     n_inputs=2,
    ...     n_outputs=3,
    ...     step=0.5,
    ...     verbose=False
    ... )
    >>> kohonet.train(X, epochs=100)
    >>> kohonet.predict(X)
    array([[ 0.,  1.,  0.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  0.,  1.]])
    """
    def predict_raw(self, X):
        X = format_data(X)
        return X.dot(self.weight)

    def predict(self, X):
        raw_output = self.predict_raw(X)
        output = np.zeros(raw_output.shape, dtype=np.int0)
        max_args = raw_output.argmax(axis=1)
        output[range(raw_output.shape[0]), max_args] = 1
        return output

    def one_training_update(self, X_train, y_train):
        step = self.step
        predict = self.predict

        error = 0
        for input_row in X_train:
            input_row = np.reshape(input_row, (1, input_row.size))
            layer_output = predict(input_row)

            _, index_y = np.nonzero(layer_output)
            distance = input_row.T - self.weight[:, index_y]
            self.weight[:, index_y] += step * distance

            error += np.abs(distance).mean()

        return error / len(X_train)
