__all__ = ('extract_network',)


def extract_network(instance):
    """
    Extract connection from different types of object.

    Parameters
    ----------
    instance : network, connection, list or tuple

    Returns
    -------
    connection

    Raises
    ------
    ValueError
        In case if input object doesn't have connection of layers.
    """
    # Note: Import it here in order to import loops
    from neupy.algorithms.base import BaseNetwork
    from neupy.layers.base import BaseGraph
    from neupy import layers

    if isinstance(instance, (list, tuple)):
        return layers.join(*instance)

    if isinstance(instance, BaseNetwork):
        return instance.network

    if isinstance(instance, BaseGraph):
        return instance

    raise TypeError(
        "Invalid input type. Input should be network, connection "
        "or list of layers, got {}".format(type(instance)))
