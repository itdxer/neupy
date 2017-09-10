import numpy as np

from .base import BaseAssociative


__all__ = ('Kohonen',)


class Kohonen(BaseAssociative):
    """
    Kohonen Neural Network used for unsupervised learning.

    Parameters
    ----------
    {BaseAssociative.n_inputs}

    {BaseAssociative.n_outputs}

    {BaseAssociative.weight}

    {BaseNetwork.step}

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.epoch_end_signal}

    {BaseNetwork.train_end_signal}

    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}

    {BaseAssociative.train}

    {BaseSkeleton.fit}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms, layers, environment
    >>>
    >>> environment.reproducible()
    >>>
    >>> input_data = np.array([
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
    >>> kohonet.train(input_data, epochs=100)
    >>> kohonet.predict(input_data)
    array([[ 0.,  1.,  0.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  0.,  1.]])
    """
    def predict_raw(self, input_data):
        input_data = self.format_input_data(input_data)
        return input_data.dot(self.weight)

    def predict(self, input_data):
        raw_output = self.predict_raw(input_data)
        output = np.zeros(raw_output.shape, dtype=np.int0)
        max_args = raw_output.argmax(axis=1)
        output[range(raw_output.shape[0]), max_args] = 1
        return output

    def train_epoch(self, input_train, target_train):
        step = self.step
        predict = self.predict

        error = 0
        for input_row in input_train:
            input_row = np.reshape(input_row, (1, input_row.size))
            layer_output = predict(input_row)

            _, index_y = np.nonzero(layer_output)
            distance = input_row.T - self.weight[:, index_y]
            self.weight[:, index_y] += step * distance

            error += np.abs(distance).mean()

        return error / len(input_train)
