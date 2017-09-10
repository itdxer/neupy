from neupy import layers
from neupy.utils import all_equal
from neupy.layers.utils import extract_connection


__all__ = ('mixture_of_experts',)


def check_if_connection_is_valid(connection, index):
    if len(connection.input_layers) > 1:
        raise ValueError(
            "Network #{} (0-based indeces) has more than one input "
            "layer. Input layers: {!r}"
            "".format(index, connection.input_layers))

    if len(connection.output_layers) > 1:
        raise ValueError(
            "Network #{} (0-based indeces) has more than one output "
            "layer. Output layers: {!r}"
            "".format(index, connection.output_layers))

    if len(connection.input_shape) != 1:
        raise ValueError(
            "Network #{} (0-based indeces) should receive vector as "
            "an input. Input layer shape: {!r}"
            "".format(index, connection.input_shape))


def check_if_connections_compatible(connections):
    input_shapes = []
    output_shapes = []

    for i, connection in enumerate(connections):
        input_shapes.append(connection.input_shape)
        output_shapes.append(connection.output_shape)

    if not all_equal(input_shapes):
        raise ValueError("Networks have different input shapes: {}"
                         "".format(input_shapes))

    if not all_equal(output_shapes):
        raise ValueError("Networks have different output shapes: {}"
                         "".format(output_shapes))


def check_if_gating_layer_valid(gating_layer, n_layers_to_combine):
    if not isinstance(gating_layer, layers.BaseLayer):
        raise ValueError(
            "Invalid type for gating layer. Type: {}"
            "".format(type(gating_layer)))

    output_shape = gating_layer.output_shape[0]
    if output_shape != n_layers_to_combine:
        raise ValueError(
            "Gating layer has invalid number of outputs. Expected {}, got {}"
            "".format(output_shape, n_layers_to_combine))


def mixture_of_experts(networks, gating_layer=None):
    """
    Generates mixture of experts architecture from the set of
    networks that has the same input and output shapes.

    Mixture of experts learns to how to mix results from different
    networks in order to get better performances. It adds gating layer
    that using input data tries to figure out which of the networks
    will make better contribution to the final result. The final result
    mixes from all networks using different weights. The higher the weight
    the larger contribution from the individual layer.

    Parameters
    ----------
    networks : list of connections or networks
        These networks will be combine into mixture of experts.
        Every network should have single 1D input layer and
        single output layer. Another restriction is that all networks
        should expect the same input and output layers.

    gating_layer : None or layer
        In case if value equal to `None` that the following layer
        will be created.

        .. code-block:: python

            gating_layer = layers.Softmax(len(networks))

        Output from the gating layer should be 1D and equal to
        the number of networks.

    Raises
    ------
    ValueError
        In case if there is some problem with input networks
        or custom gating layer.

    Returns
    -------
    connection
        Mixture of experts network that combine all networks into
        single one and adds gating layer to it.

    Examples
    --------
    >>> from neupy import layers, algorithms, architectures
    >>>
    >>> network = architectures.mixture_of_experts([
    ...     layers.join(
    ...         layers.Input(10),
    ...         layers.Relu(5),
    ...     ),
    ...     layers.join(
    ...         layers.Input(10),
    ...         layers.Relu(33),
    ...         layers.Relu(5),
    ...     ),
    ...     layers.join(
    ...         layers.Input(10),
    ...         layers.Relu(12),
    ...         layers.Relu(25),
    ...         layers.Relu(5),
    ...     ),
    ... ])
    >>> network
    10 -> [... 12 layers ...] -> 5
    >>>
    >>> gdnet = algorithms.Momentum(network, step=0.1)
    """
    if not isinstance(networks, (list, tuple)):
        raise ValueError("Networks should be specified as a list")

    connections = []
    for index, network in enumerate(networks):
        connection = extract_connection(network)
        check_if_connection_is_valid(connection, index)
        connections.append(connection)

    check_if_connections_compatible(connections)

    first_connection = connections[0]
    n_features = first_connection.input_shape[0]
    n_layers_to_combine = len(connections)

    if gating_layer is None:
        gating_layer = layers.Softmax(n_layers_to_combine)

    check_if_gating_layer_valid(gating_layer, n_layers_to_combine)

    return layers.join(
        layers.Input(n_features),
        # Note: Gating network should be specified
        # as a first parameter.
        [gating_layer] + connections,
        layers.GatedAverage())
