from functools import partial
from collections import defaultdict

import six
from six.moves import cPickle as pickle

from neupy.utils import asfloat
from neupy.algorithms.base import BaseNetwork
from neupy.layers.utils import iter_parameters


__all__ = ('save', 'load')


iter_parameters = partial(iter_parameters, only_trainable=False)


def save(connection, filepath):
    """
    Save layer parameters in pickle file.

    Parameters
    ----------
    connection : network, list of layer or connection
        Connection that needs to be saved.

    filepath : str
        Path to the pickle file that will store
        network's parameters.
    """
    if isinstance(connection, BaseNetwork):
        connection = connection.connection

    data = defaultdict(dict)

    for layer, attrname, parameter in iter_parameters(connection):
        data[layer.name][attrname] = parameter.get_value()

    with open(filepath, 'wb+') as f:
        pickle.dump(data, f)


def load(connection, source, ignore_missed=False):
    """
    Load and set parameters for layers from the
    specified source.

    Parameters
    ----------
    connection : list of layers or connection

    source : str or dict
        It can be path to the pickle file that stores
        parameters or dictionary that has key values that
        store layer name and values is a dictionary that
        stores parameter names and their values.

    ignore_missed : bool
        ``False`` means that error will be triggered in case
        if some of the layers doesn't have storage parameters
        in the specified source. Defaults to ``False``.

    Raises
    ------
    TypeError
        In case if source has invalid data type.
    """
    if isinstance(connection, BaseNetwork):
        connection = connection.connection

    if isinstance(source, six.string_types):
        with open(source, 'rb') as f:
            data = pickle.load(f)

    elif isinstance(source, dict):
        data = source

    else:
        raise TypeError("Source type is unknown. Got {}, expected dict "
                        "or str".format(type(source)))

    for layer, attrname, _ in iter_parameters(connection):
        if layer.name not in data or attrname not in data[layer.name]:
            if ignore_missed:
                continue

            raise ValueError("Cannot load parameters from the specified "
                             "data source. Layer `{}` doesn't have "
                             "stored parameter `{}`."
                             "".format(layer.name, attrname))

        loaded_parameter = data[layer.name][attrname]

        attrvalue = getattr(layer, attrname)
        attrvalue.set_value(asfloat(loaded_parameter))

    # We need to initalize connection, to make sure
    # that each layer will generate shared variables
    # and validate connections
    connection.initialize()
