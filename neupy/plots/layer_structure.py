import copy
import tempfile
from collections import OrderedDict

import graphviz

from neupy.layers.base import ResidualConnection
from neupy.algorithms.base import BaseNetwork


__all__ = ('layer_structure',)


def layer_uid(layer):
    """
    Gets unique identifier (UID) for the layer.

    Parameters
    ----------
    layer : layer

    Returns
    -------
    str
    """
    return str(id(layer))


def format_label(info):
    """
    Format label information.

    Parameters
    ----------
    object

    Returns
    -------
    str
    """
    # Space at the beggining shifts string
    # to the right
    return " {}".format(info)


def exclude_layer_from_graph(graph, ignore_layers):
    """
    Exclude specific types of layers from the graph.

    Parameters
    ----------
    graph : dict
        Layer graph where each value is a list of layers
        defined in this graph as a key.

    ignore_layers : list or tuple of layers
        Layer types that need to be exclude from the graph.

    Returns
    -------
    dict
        New graph with excluded layers.
    """
    ignore_layers = tuple(ignore_layers)
    cleaned_graph = OrderedDict()

    for from_layer, to_layers in graph.items():
        if isinstance(from_layer, ignore_layers):
            continue

        cleaned_graph[from_layer] = []
        to_layers = copy.copy(to_layers)

        while to_layers:
            to_layer = to_layers.pop()

            if isinstance(to_layer, ignore_layers):
                # from_layer connects to the to_layer that
                # we need to exclude. For this reason we just
                # connect from_layer to layers that connected
                # from the to_layer
                #
                # Before:
                # from_layer -> to_layer -> next_to_layers
                #
                # After:
                # from_layer-> next_to_layers
                to_layers.extend(graph[to_layer])

            else:
                cleaned_graph[from_layer].append(to_layer)

    return cleaned_graph


def layer_structure(connection, ignore_layers=None, filepath=None, show=True):
    """
    Draw graphical representation of the layer connection
    structure in form of directional graph.

    Parameters
    ----------
    connection : BaseLayer instance, BaseNetwork instance

    ignore_layers : list or None
        List of layer types that needs to be excluded
        from the plot. Defaults to ``None``.

    filepath : str or None
        Path to the file that stores graph. ``None`` means
        that file will be saved in temporary file.
        Defaults to ``None``.

    show : bool
        ``True`` opens PDF file. Defaults to ``True``.

    Examples
    --------
    >>> from neupy import layers, plots
    >>>
    >>> connection = layers.Input(10) > layers.Sigmoid(1)
    >>> plots.layer_structure(connection)
    """
    if isinstance(connection, BaseNetwork):
        connection = connection.connection

    if ignore_layers is None:
        ignore_layers = []

    if filepath is None:
        filepath = tempfile.mktemp()

    ignore_layers = [ResidualConnection] + ignore_layers

    forward_graph = connection.graph.forward_graph
    forward_graph = exclude_layer_from_graph(forward_graph, ignore_layers)

    digraph = graphviz.Digraph()

    for layer in forward_graph.keys():
        digraph.node(layer_uid(layer), str(layer))

    output_id = 1
    for from_layer, to_layers in forward_graph.items():
        for to_layer in to_layers:
            digraph.edge(layer_uid(from_layer), layer_uid(to_layer),
                         label=format_label(from_layer.output_shape))

        if not to_layers:
            output = 'output-{}'.format(output_id)

            digraph.node(output, 'Output #{}'.format(output_id))
            digraph.edge(layer_uid(from_layer), output,
                         label=" {}".format(from_layer.output_shape))

            output_id += 1

    digraph.render(filepath, view=show)
