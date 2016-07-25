from copy import deepcopy
from functools import reduce

from neupy.network import ConstructableNetwork
from neupy.layers.connections import LayerConnection
from neupy import layers


__all__ = ('cut', 'sew_together', 'CutLine', 'cut_along_lines',
           'isolate_connection_if_needed')


def isolate_connection(connection):
    """
    Withdraw previous connections.

    Parameters
    ----------
    connection : LayerConnection instance
        Connection that you need to isolate.
    """
    connection.input_layer.relate_from_layer = None
    connection.input_layer.connection = None

    connection.output_layer.relate_to_layer = None
    connection.output_layer.connection = None

    connection.connection = None


def isolate_layer(layer):
    """
    Withdraw previous connections.

    Parameters
    ----------
    layer : BaseLayer instance
        Layer that you need to isolate.
    """
    layer.relate_from_layer = None
    layer.connection = None
    layer.relate_to_layer = None


def is_layer_isolated(connection):
    """
    Check whether connection is not depend on the
    other connections.

    Parameters
    ----------
    connection : BaseLayer instance
        Layer or connection that you need to validate.

    Returns
    -------
    bool
        ``True`` means that connection or layer is independent.
        ``False`` means that instance is related to some other
        layers.
    """
    is_not_relate_from_sb = (connection.relate_from_layer is None)
    is_not_relate_to_sb = (connection.relate_to_layer is None)
    return (is_not_relate_from_sb and is_not_relate_to_sb)


def is_connection_isolated(connection):
    """
    Check whether connection is not depend on the
    other connections.

    Parameters
    ----------
    connection : LayerConnection or BaseLayer instance
        Layer or connection that you need validate.

    Returns
    -------
    bool
        ``True`` means that connection or layer is independent.
        ``False`` means that instance is related to some other
        layers.
    """
    is_not_relate_from_sb = (connection.input_layer.relate_from_layer is None)
    is_not_relate_to_sb = (connection.output_layer.relate_to_layer is None)
    return (is_not_relate_from_sb and is_not_relate_to_sb)


def isolate_connection_if_needed(connection):
    """
    Function copies connection or layer if one is related
    to the other layers. In addition it gets rid of these
    connections.

    Parameters
    ----------
    connection : BaseLayer or LayerConnection instance
        Layer or combined layers that needs to be validated.

    Returns
    -------
    BaseLayer or LayerConnection instance
        Copy of the object or the same instance.

    Raises
    ------
    ValueError
        If input data type is incorrect.
    """
    is_layer = isinstance(connection, layers.BaseLayer)
    is_connection = isinstance(connection, LayerConnection)

    if is_layer and not is_layer_isolated(connection):
        connection = deepcopy(connection)
        isolate_layer(connection)

    elif is_connection and not is_connection_isolated(connection):
        connection = deepcopy(connection)
        isolate_connection(connection)

    elif not is_layer and not is_connection:
        raise TypeError("Unknown data type: {}. Surgery module supports "
                        "only procedures with layers and connections "
                        "between layers.".format(type(connection)))

    return connection


def clean_and_validate_connection(connection):
    """

    Parameters
    ----------
    connection : ConstructableNetwork ot LayerConnection instance
        Network class that has constructuble layers or
        connected layers.

    Returns
    -------
    LayerConnection
    """
    if isinstance(connection, ConstructableNetwork):
        # Re-define variable to make it easy to understand that
        # object in not a real connection.
        # The two lines below looks more information that just
        # write it as ``connection = connection.connection``
        network = connection
        connection = network.connection

    if not isinstance(connection, LayerConnection):
        raise ValueError("You can cut only layer connections.")

    return connection


def cut(connection, start, end):
    """
    Function cuts a specific part of the neural networks
    structure. Function works in the same way as a slicing in
    Python. You can think about it as a ``layers[start:end]``.

    Parameters
    ----------
    connection : ConstructableNetwork ot LayerConnection instance
        Network class that has constructuble layers or
        connected layers.
    start : int
        Index of the first layer in the new sequence.
    end : int
        Index of the final layer in the new sequence.

    Returns
    -------
    LayerConnection instance
        Redefined connection between cutted layers.

    Raises
    ------
    ValueError
        In case if something is wrong with the input parameters.

    Examples
    --------
    >>> from neupy import layers, surgery
    >>> layers = [
    ...     layers.Input(10),
    ...     layers.Sigmoid(20),
    ...     layers.Sigmoid(30),
    ...     layers.Sigmoid(40),
    ... ]
    >>> connection = surgery.sew_together(layers)
    >>> connection
    Input(10) > Sigmoid(20) > Sigmoid(30) > Sigmoid(40)
    >>>
    >>> surgery.cut(connection, start=1, end=3)
    Sigmoid(20) > Sigmoid(30)
    """
    connection = clean_and_validate_connection(connection)
    layers = list(connection)
    n_layers = len(layers)

    if end > n_layers:
        raise ValueError("Cannot cut till the {} layer. Connection has "
                         "only {} layers.".format(end, n_layers))

    cutted_layers = layers[start:end]

    if not cutted_layers:
        raise ValueError("Your slice didn't cut any layer.")

    return sew_together(cutted_layers)


def sew_together(connections):
    """
    Connect layers and connections together.

    Parameters
    ----------
    connections : list
        List of layers and layer connections.

    Returns
    -------
    BaseLayer instance, LayerConnection instance or None
        Combined layers and partial connections in one
        bug connection. ``None`` result means that your input
        is an empty list or tuple. If you get a layer instead
        of connection it mean that you have just one layer in the
        sequence.

    Examples
    --------
    >>> from neupy import layers, surgery
    >>> connection = surgery.sew_together([
    ...    layers.Input(784),
    ...    layers.Relu(30) > layers.Relu(20),
    ...    layers.Softmax(10),
    ... ])
    >>> connection
    Input(784) > Relu(30) > Relu(20) > Softmax(10)
    """
    if not connections:
        return

    cleaned_connections = []
    for connection in connections:
        # Since connection can be related to some other network,
        # we need to copy it and clean all old relations.
        # It will help us to prevent errors in future.
        connection = isolate_connection_if_needed(connection)
        cleaned_connections.append(connection)

    merged_connections = reduce(LayerConnection, cleaned_connections)
    return merged_connections


class CutLine(layers.BaseLayer):
    """
    Basic layer class that doesn't change network's structure.
    This class just help you to define places where you need to
    split your layer's structure.
    """


def iter_cutted_regions(cutted_regions):
    """
    Takes a list of integer and iterates over non-empty
    slicing index pairs.

    Parameters
    ----------
    cutted_regions : list of int
        List of indeces that defines cut points.

    Yields
    ------
    tuple with 2 int
        It contains indeces that defines cut points
        ``(left_index, right_index)``.

    Examples
    --------
    >>> from neupy import surgery
    >>> regions = [0, 1, 5, 7, 9, 10]
    >>> for start, end in surgery.iter_cutted_regions(regions):
    ...     print(start, end)
    ...
    1 4
    5 6
    7 8
    """
    left_bounds = cutted_regions
    right_bounds = cutted_regions[1:]

    for left_index, right_index in zip(left_bounds, right_bounds):
        # We try to avoid cases when cutted region gives an empty slice.
        if right_index - left_index > 1:
            yield (left_index, right_index - 1)


def find_cut_points(connection):
    """
    Function looks for the cut lines in the connection.

    Parameters
    ----------
    connection : LayerConnection instance
        Connected layers.

    Returns
    -------
    list of int
        The final result is a list of indeces that defines cut
        line layers position. One important note is that function
        automaticaly added the first and the last layers as a cut
        points. It means that you will always get at least
        two values in the list.

    Notes
    -----
    It's better to use ``surgery.cut_along_lines`` function to cut
    your layers.

    Examples
    --------
    >>> from neupy import layers, surgery
    >>> connection = layers.Input(10) > layers.Sigmoid(5)
    >>> connection
    Input(10) > Sigmoid(5)
    >>> surgery.find_cut_points(connection)
    [0, 3]
    """
    n_layers = len(connection)
    # We assume that first index is always a 'cut line', that's
    # wht we added 0 as a first index. We don't connection them
    # directly at the beginning to prevent all issues related
    # to the connection structure modifications
    cut_points = [0]

    # Since we 'added' cut line layer at the beginning, we need
    # to start count other layers from first index
    for i, layer in enumerate(connection, start=1):
        if isinstance(layer, CutLine):
            cut_points.append(i)

    # We also assume that we have a final layer as a 'cut line'.
    # And again we aren't adding layer directly. We just
    # assuming that we have it in the network
    cut_points.append(n_layers + 1)
    return cut_points


def cut_along_lines(connection):
    """
    Cuts layer's connection in the specified places.
    in the places where you need to cut layer you need to set up
    ``surgery.CutLine`` layer, that defines place where you need
    to cut the network.

    Parameters
    ----------
    connection : ConstructableNetwork ot LayerConnection instance
        Network class that has constructuble layers or
        connected layers.

    Returns
    -------
    list
        List that contains all cutted connections.

    Examples
    --------
    >>> from neupy import algorithms, layers, surgery
    >>> network = algorithms.GradientDescent([
    ...     layers.Input(5),
    ...     surgery.CutLine(),  # <- first cut point
    ...     layers.Sigmoid(10),
    ...     layers.Sigmoid(20),
    ...     layers.Sigmoid(30),
    ...     surgery.CutLine(),  # <- second cut point
    ...     layers.Sigmoid(1),
    ... ])
    >>> cutted_connections = surgery.cut_along_lines(network)
    >>>
    >>> for connection in cutted_connections:
    ...     print(connection)
    ...
    Input(5)
    Sigmoid(10) > Sigmoid(20) > Sigmoid(30)
    Sigmoid(1)
    """
    connection = clean_and_validate_connection(connection)
    cut_points = find_cut_points(connection)

    connections = []
    for start, end in iter_cutted_regions(cut_points):
        cutted_connection = cut(connection, start, end)
        connections.append(cutted_connection)

    return connections
