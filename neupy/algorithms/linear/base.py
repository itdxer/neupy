from neupy.layers import Step, Input
from neupy.layers.connections import LayerConnection
from neupy.exceptions import InvalidConnection
from neupy.algorithms.constructor import ConstructibleNetwork


__all__ = ('BaseLinearNetwork',)


class BaseLinearNetwork(ConstructibleNetwork):
    """
    Base class for feedforward neural network without hidden layers.

    Notes
    -----
    - Input layer should be :layer:`Step` class instance.

    - If you need to define specific weights for the
      network then you should initialize connection

      .. code-block:: python

          import numpy as np
          from neupy.layers import Input, Step

          connection = Input(5) > Step(1, weight=np.ones((5, 1)))

    Parameters
    ----------
    connection : list, tuple or LayerConnection instance
        Should be a list or tuple that contains two integers.
        First integer defines number of input units and the
        seconds one number of output units.

    {ConstructibleNetwork.error}

    {BaseNetwork.Parameters}

    Methods
    -------
    {BaseSkeleton.predict}

    {ConstructibleNetwork.train}

    {BaseSkeleton.fit}
    """
    def __init__(self, connection, **options):
        if len(connection) != 2:
            raise ValueError("This network should contains two layers.")

        if all(isinstance(element, int) for element in connection):
            input_layer_size, output_layer_size = connection
            connection = Input(input_layer_size) > Step(output_layer_size)

        if not isinstance(connection, LayerConnection):
            raise ValueError("Invalid connection type")

        output_layer = connection.output_layers[0]

        if not isinstance(output_layer, Step):
            raise InvalidConnection(
                "Final layer should contains step activation function "
                "(``layers.Step`` class instance)."
            )

        super(BaseLinearNetwork, self).__init__(connection, **options)

    def init_variables(self):
        super(BaseLinearNetwork, self).init_variables()
        self.variables.network_input = self.variables.network_inputs[0]
