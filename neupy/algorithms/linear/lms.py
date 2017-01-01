from neupy.algorithms.linear.base import BaseLinearNetwork


__all__ = ('LMS',)


class LMS(BaseLinearNetwork):
    """
    LMS Neural Network. Algorithm has several names,
    including the Widrow-Hoff or Delta rule.

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
    >>> target_data = np.array([[1], [0], [0], [1]])
    >>>
    >>> lmsnet = algorithms.LMS((2, 1), step=0.5)
    >>>
    >>> lmsnet.train(input_data, target_data, epochs=200)
    >>> lmsnet.predict(np.array([[4, 4], [0, 0]]))
    array([[0],
           [1]])

    See Also
    --------
    :network:`Perceptron` : Perceptron Neural Network.
    """

    def init_layer_updates(self, layer):
        if not layer.parameters:
            return []

        network_output = self.variables.network_output
        network_input = self.variables.network_input
        step = self.variables.step

        summated_output = network_input.dot(layer.weight) + layer.bias
        linear_error = summated_output - network_output

        normalized_input = network_input / network_input.norm(L=2)
        weight_delta = normalized_input.T.dot(linear_error)
        bias_delta = linear_error.sum(axis=0)

        return [
            (layer.weight, layer.weight - step * weight_delta),
            (layer.bias, layer.bias - step * bias_delta),
        ]
