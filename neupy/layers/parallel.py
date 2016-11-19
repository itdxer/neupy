from .utils import join as layers_join
from .base import BaseLayer


__all__ = ('Parallel', 'parallel')


class TransferLayer(BaseLayer):
    """
    Hack layer for parallel connections.
    """


def parallel(connections, merge_layer):
    """
    Propagate input value through the multiple parallel layer
    connections and then combine output result.

    Parameters
    ----------
    connections : list of lists
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
    >>> parallel_layer = layers.parallel(
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

    if not isinstance(merge_layer, BaseLayer):
        raise ValueError("The `merge_layer` argument is not an instance of "
                         "BaseLayer class.")

    input_layer = TransferLayer()

    for i, connection in enumerate(connections):
        if isinstance(connection, (list, tuple)):
            connection = layers_join(connection)
            connections[i] = connection

        full_connection = layers_join(input_layer, connection, merge_layer)

    return full_connection


Parallel = parallel  # alias
