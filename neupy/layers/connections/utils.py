from itertools import chain
from functools import reduce


def join(*connections):
    """
    Connect two layers.

    Parameters
    ----------
    *connections : layers or connections

    Returns
    -------
    connection
        Layers connected in a sequence.

    Examples
    --------
    >>> from neupy import layers
    >>> conn = layers.join(
    ...     layers.Input(784),
    ...     layers.Relu(500),
    ...     layers.Relu(300),
    ...     layers.Softmax(10),
    ... )
    """
    from .base import LayerConnection, ParallelConnection

    if not connections:
        return

    if len(connections) == 1:
        connection = connections[0]

        if isinstance(connection, (list, tuple)):
            return ParallelConnection(connection)

        return connection

    merged_connections = reduce(LayerConnection, connections)
    return merged_connections


def is_sequential(connection):
    """
    Check whether graph connection is a sequence.

    Parameters
    ----------
    connection : connection

    Returns
    -------
    bool
    """
    forward_graph_layers = connection.graph.forward_graph.values()
    backward_graph_layers = connection.graph.backward_graph.values()

    for layers in chain(forward_graph_layers, backward_graph_layers):
        if len(layers) >= 2:
            # One of the layers has multiple input
            # or output connections
            return False

    return True
