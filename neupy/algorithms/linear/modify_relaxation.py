import theano.tensor as T

from neupy.core.properties import BoundedProperty
from neupy.algorithms.linear.base import BaseLinearNetwork


__all__ = ('ModifiedRelaxation',)


class ModifiedRelaxation(BaseLinearNetwork):
    """
    Modified Relaxation Neural Network. Simple linear network. If the
    output value of the network received more than the set limit, the
    weight is updated in the same way as the :network:`LMS`, if less
    than the set value - the update will be in proportion to the
    expected result.

    Parameters
    ----------
    dead_zone_radius : float
        Indicates the line between stable outcome network output and
        weak, and depending on the result of doing different updates.

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
    >>> mrnet = algorithms.ModifiedRelaxation((2, 1), step=1)
    >>>
    >>> mrnet.train(input_data, target_data, epochs=100)
    >>> mrnet.predict(np.array([[4, 4], [0, 0]]))
    array([[0],
           [1]])

    See Also
    --------
    :network:`LMS` : LMS Neural Network.
    """
    dead_zone_radius = BoundedProperty(default=0.1, minval=0)

    def init_train_updates(self):
        layer = self.connection.output_layers[0]

        prediction_func = self.variables.train_prediction_func
        network_output = self.variables.network_output
        network_input = self.variables.network_input
        step = self.variables.step

        normalized_input = network_input / network_input.norm(L=2)
        summated_output = network_input.dot(layer.weight) + layer.bias
        linear_error = prediction_func - network_output
        update = T.where(
            T.abs_(summated_output) >= self.dead_zone_radius,
            linear_error,
            network_output
        )

        weight_delta = normalized_input.T.dot(update)
        bias_delta = linear_error.sum(axis=0)

        return [
            (layer.weight, layer.weight - step * weight_delta),
            (layer.bias, layer.bias - step * bias_delta),
        ]
