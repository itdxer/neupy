from .connections import ChainConnection
from .utils import join as layers_join
from .base import BaseLayer


__all__ = ('Parallel',)


class TransferLayer(BaseLayer):
    """
    Hack for parallel connections.
    """


def Parallel(connections, merge_layer):
    """
    Propagate input value through the multiple parallel layer
    connections and then combine output result.

    Parameters
    ----------
    connections : list of lists, list of LayerConnection
        List that contains list of layer connections.

    merge_layer : BaseLayer instance
        Layer that merges final outputs from each parallel
        connection.

    Returns
    -------
    LayerConnection

    Examples
    --------
    >>> from neupy import layers
    >>>
    >>> parallel_layer = layers.Parallel(
    ...     [[
    ...         layers.Convolution((3, 5, 5)),
    ...     ], [
    ...         layers.Convolution((10, 3, 3)),
    ...         layers.Convolution((5, 3, 3)),
    ...     ]],
    ...     layers.Concatenate()
    ... )
    """
    if not isinstance(connections, (list, tuple)):
        raise ValueError("Connections should be a list or a tuple.")

    if not isinstance(merge_layer, ChainConnection):
        raise ValueError("The `merge_layer` argument is not "
                         "a layer or connection")

    input_layer = TransferLayer()

    for i, connection in enumerate(connections):
        if not connection:
            full_connection = layers_join(input_layer, merge_layer)
            continue

        if isinstance(connection, (list, tuple)):
            connection = layers_join(connection)
            connections[i] = connection

        full_connection = layers_join(input_layer, connection, merge_layer)

    return full_connection
