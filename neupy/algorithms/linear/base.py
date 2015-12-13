from neupy.utils import is_list_of_integers
from neupy.layers.connections import NetworkConnectionError, LayerConnection
from neupy.network import SupervisedConstructableNetwork
from neupy.layers import Step, Output


__all__ = ('BaseLinearNetwork',)


class BaseLinearNetwork(SupervisedConstructableNetwork):
    """ Base class for feedforward neural network without hidden layers.

    Notes
    -----
    * Input layer should be :layer:`Step` class instance.

    Parameters
    ----------
    connection : list, tuple or object
        Should be a list or tuple that contains two integers. First integer
        describe number of input units and the seconds one number of output
        units.
    {SupervisedConstructableNetwork.error}
    {BaseNetwork.step}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Methods
    -------
    {BaseSkeleton.predict}
    {SupervisedLearning.train}
    {BaseSkeleton.fit}
    {BaseNetwork.plot_errors}
    {BaseNetwork.last_error}
    {BaseNetwork.last_validation_error}
    {BaseNetwork.previous_error}
    """

    def __init__(self, connection, **options):
        if len(connection) != 2:
            raise ValueError("This network should contains two layers.")

        if is_list_of_integers(connection):
            input_layer_size, output_layer_size = connection
            connection = Step(input_layer_size) > Output(output_layer_size)

        if not isinstance(connection, LayerConnection):
            raise ValueError("Invalid network connection structure.")

        if not isinstance(connection.input_layer, Step):
            raise NetworkConnectionError(
                "Input layer should contains step activation function "
                "(``Step`` class instance)."
            )

        super(BaseLinearNetwork, self).__init__(connection, **options)
