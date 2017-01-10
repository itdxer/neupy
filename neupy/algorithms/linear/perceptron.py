from neupy.algorithms.linear.base import BaseLinearNetwork


__all__ = ('Perceptron',)


class Perceptron(BaseLinearNetwork):
    """
    Perceptron Neural Network. Simples linear model in Neural
    Networks.

    Notes
    -----
    {BaseLinearNetwork.Notes}

    Parameters
    ----------
    {BaseLinearNetwork.connection}

    {BaseNetwork.step}

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.epoch_end_signal}

    {BaseNetwork.train_end_signal}

    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}

    {ConstructibleNetwork.train}

    {BaseSkeleton.fit}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> input_data = np.array([[1, 0], [2, 2], [3, 3], [0, 0]])
    >>> target_data = np.array([[1], [-1], [-1], [1]])
    >>>
    >>> mrnet = algorithms.Perceptron((2, 1), step=0.4)
    >>>
    >>> mrnet.train(input_data, target_data, epochs=30)
    >>> mrnet.predict(np.array([[4, 4], [-1, -1]]))
    array([[-1],
           [ 1]])
    """
    def init_train_updates(self):
        layer = self.connection.output_layers[0]

        prediction_func = self.variables.prediction_func
        network_output = self.variables.network_output
        network_input = self.variables.network_input
        step = self.variables.step

        linear_error = prediction_func - network_output
        weight_delta = network_input.T.dot(linear_error)
        bias_delta = linear_error.sum(axis=0)

        return [
            (layer.weight, layer.weight - step * weight_delta),
            (layer.bias, layer.bias - step * bias_delta),
        ]
