import numpy as np

from neupy.core.properties import NonNegativeNumberProperty
from neupy.algorithms.basics.base import SimpleTwoLayerNetwork


__all__ = ('ModifiedRelaxation',)


class ModifiedRelaxation(SimpleTwoLayerNetwork):
    """ Modified Relaxation Neural Network. Simple linear network. If the
    output value of the network received more than the set limit, the
    weight is updated in the same way as the :network:`LMS`, if less
    than the set value - the update will be in proportion to the
    expected result.

    Parameters
    ----------
    dead_zone_radius : float
        Indicates the line between stable outcome network output and
        weak, and depending on the result of doing different updates.
    {full_params}

    Methods
    -------
    {supervised_train}
    {full_methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> input_data = np.array([[1, 0], [2, 2], [3, 3], [0, 0]])
    >>> target_data = np.array([[1], [-1], [-1], [1]])
    >>>
    >>> mrnet = algorithms.ModifiedRelaxation((2, 1), step=1, verbose=False)
    >>> mrnet.train(input_data, target_data, epochs=100)
    >>> mrnet.predict(np.array([[4, 4], [-1, -1]]))
    array([[-1],
           [ 1]])

    See Also
    --------
    :network:`LMS` : LMS Neural Network.
    """
    dead_zone_radius = NonNegativeNumberProperty(default=0.1)

    def get_weight_delta(self, output_train, target_train):
        input_data = self.input_data
        update = np.where(
            np.abs(self.summated) >= self.dead_zone_radius,
            self.error(output_train, target_train),
            target_train
        )
        minimized_input = input_data / np.linalg.norm(input_data) ** 2

        return np.dot(minimized_input.T, update)
