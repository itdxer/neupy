from neupy.utils import is_list_of_integers
from neupy.layers.connections import NetworkConnectionError, LayerConnection
from neupy.network import ConstructableNetwork
from neupy.layers import Step, Input


__all__ = ('BaseLinearNetwork',)


class BaseLinearNetwork(ConstructableNetwork):
    """
    Base class for feedforward neural network without hidden layers.

    Notes
    -----
    * Input layer should be :layer:`Step` class instance.

    Parameters
    ----------
    connection : list, tuple or object
        Should be a list or tuple that contains two integers. First integer
        describe number of input units and the seconds one number of output
        units.
    {ConstructableNetwork.error}
    {BaseNetwork.step}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}
    {SupervisedLearningMixin.train}
    {BaseSkeleton.fit}
    """

    def __init__(self, connection, **options):
        if len(connection) != 2:
            raise ValueError("This network should contains two layers.")

        if is_list_of_integers(connection):
            input_layer_size, output_layer_size = connection
            connection = Input(input_layer_size) > Step(output_layer_size)

        if not isinstance(connection, LayerConnection):
            raise ValueError("Invalid network connection structure.")

        if not isinstance(connection.output_layer, Step):
            raise NetworkConnectionError(
                "Final layer should contains step activation function "
                "(``layers.Step`` class instance)."
            )

        super(BaseLinearNetwork, self).__init__(connection, **options)
