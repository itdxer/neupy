import tempfile

import graphviz

from neupy.network.base import BaseNetwork


__all__ = ('layer_structure',)


def layer_uid(layer):
    """
    Gets unique identifier (UID) for the layer.

    Parameters
    ----------
    layer : BaseLayer instance

    Returns
    -------
    str
    """
    return str(id(layer))


def layer_structure(connection, filepath=None, show=True):
    """
    Draw graphical representation of the layer connection
    structure in form of directional graph.

    Parameters
    ----------
    connection : BaseLayer instance, BaseNetwork instance
    filepath : str or None
        Path to the file that stores graph. ``None`` means that file
        will be saved in temporary file. Defaults to ``None``.
    show : bool
        ``True`` opens PDF file. Defaults to ``True``.

    Raises
    ------
    ImportError
        In case if ``graphviz`` library hasn't been installed.
    """
    if isinstance(connection, BaseNetwork):
        connection = connection.connection

    if connection.graph is None:
        return

    if filepath is None:
        filepath = tempfile.mktemp()

    forward_graph = connection.graph.forward_graph
    digraph = graphviz.Digraph()

    for layer in forward_graph.keys():
        digraph.node(layer_uid(layer), str(layer))

    for from_layer, to_layers in forward_graph.items():
        for to_layer in to_layers:
            digraph.edge(layer_uid(from_layer),
                         layer_uid(to_layer),
                         # Space at the beggining shifts string
                         # to the right
                         label=" {}".format(from_layer.output_shape))

    digraph.render(filepath, view=show)
