import json
from time import gmtime, strftime

import six
import h5py
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

import neupy
from neupy.core.docs import shared_docs
from neupy.layers.graph import LayerGraph
from neupy.algorithms.base import BaseNetwork
from neupy.utils import asfloat, tf_utils


__all__ = (
    'save', 'load',  # aliases to hdf5
    'save_pickle', 'load_pickle',
    'save_json', 'load_json',
    'save_hdf5', 'load_hdf5',
    'load_dict', 'save_dict',
)


class ParameterLoaderError(Exception):
    """
    Exception triggers in case if there are some issues
    during the parameter loading procedure.
    """


class InvalidFormat(Exception):
    """
    Exception triggers when there are some issue with
    data format that stores network data.
    """


def extract_network(instance):
    if isinstance(instance, BaseNetwork):
        return instance.network

    if isinstance(instance, LayerGraph):
        return instance

    raise TypeError(
        "Invalid input type. Input should be network or optimizer "
        "with network, got `{}` instance instead".format(type(instance)))


def load_layer_parameter(layer, layer_data):
    """
    Set layer parameters to the values specified in the
    stored data
    """
    session = tf_utils.tensorflow_session()

    for param_name, param_data in layer_data['parameters'].items():
        parameter = getattr(layer, param_name)

        if not isinstance(parameter, tf.Variable):
            raise ParameterLoaderError(
                "The `{}` parameter from the `{}` layer expected to be "
                "instance of the tf.Variable, but current value equal to {}. "
                "Layer: {}".format(param_name, layer.name, parameter, layer))

        parameter.load(asfloat(param_data['value']), session)


def load_dict_by_names(layers_conn, layers_data, ignore_missing=False):
    """"
    Load parameters in to layer using layer names as the reference.

    Raises
    ------
    ParameterLoaderError
        In case if it's impossible to load parameters from data

    Returns
    -------
    bool
        Returns `True` in case if data was loaded successfully
        and `False` when parameters wasn't loaded
    """
    layers_data = {l['name']: l for l in layers_data}
    layers_conn = {l.name: l for l in layers_conn}

    if not ignore_missing and layers_data.keys() != layers_conn.keys():
        raise ParameterLoaderError(
            "Cannot match layers by name. \n"
            "  Layer names in network: {}\n"
            "  Layer names in stored data: {}"
            "".format(layers_conn.keys(), layers_data.keys()))

    elif ignore_missing and all(l not in layers_data for l in layers_conn):
        raise ParameterLoaderError("Non of the layers can be matched by name")

    for layer_name, layer in layers_conn.items():
        if layer_name in layers_data:
            load_layer_parameter(layer, layers_data[layer_name])


def load_dict_sequentially(layers_conn, layers_data):
    """"
    Load parameters in to layer using sequential order of
    layer in network and stored data
    """
    for layer, layer_data in zip(layers_conn, layers_data):
        load_layer_parameter(layer, layer_data)


def validate_data_structure(data):
    """
    Validates structure of the stored data

    Parameters
    ----------
    data : dict

    Raises
    ------
    InvalidFormat
        When format is invalid
    """
    if not isinstance(data, dict):
        raise InvalidFormat("Stored data should be in dictionary format")

    if 'layers' not in data:
        raise InvalidFormat("Stored data has no key `layers`")

    if not isinstance(data['layers'], list):
        raise InvalidFormat("Layers stored not in the list format")

    if not data['layers']:
        raise InvalidFormat("Stored data don't have any layer stored")

    for layer_index, layer in enumerate(data['layers']):
        if not isinstance(layer, dict):
            raise InvalidFormat(
                "Layer in the {} position (0-based indices) is not a "
                "dictionary (it is {})".format(layer_index, type(layer)))

        for attr in ('parameters', 'name'):
            if attr not in layer:
                raise InvalidFormat(
                    "Layer in the {} position (0-based indices) don't "
                    "have key `{}` specified".format(layer_index, attr))

        if not isinstance(layer['parameters'], dict):
            raise InvalidFormat(
                "Layer in the {} position (0-based indices) parameters "
                "specified as `{}`, but dictionary expected"
                "".format(layer_index, type(layer['parameters'])))

        for param_name, param in layer['parameters'].items():
            if not isinstance(param, dict):
                raise InvalidFormat(
                    "Layer in the {} position (0-based indices) has "
                    "incorrect value for parameter named `{}`. It has "
                    "to be a dictionary, but got {}"
                    "".format(layer_index, param_name, type(param)))

            if 'value' not in param:
                raise InvalidFormat(
                    "Layer in the {} position (0-based indices) has "
                    "incorrect value for parameter named `{}`. Parameter "
                    "doesn't have key named `value`"
                    "".format(layer_index, param_name))


def load_dict(network, data, ignore_missing=False,
              load_by='names_or_order', skip_validation=True):
    """
    Load network networks from dictionary.

    Parameters
    ----------
    network : network, list of layer or network

    data : dict
        Dictionary that stores network parameters.

    ignore_missing : bool
        ``False`` means that error will be triggered in case
        if some of the layers doesn't have storage parameters
        in the specified source. Defaults to ``False``.

    load_by : {``names``, ``order``, ``names_or_order``}
        Defines strategy that will be used during parameter loading

        - ``names`` - Matches layers using their names.

        - ``order`` - Matches layers using their relative position.

        - ``names_or_order`` - First, method tries to match layers using
          their names. If that doesn't work, then layer's relative position
          will be used.

        Defaults to ``names_or_order``.

    skip_validation : bool
        When set to ``False`` validation will be applied per each layer in
        order to make sure that there were no changes between created
        and stored models. Defaults to ``True``

    Raises
    ------
    ValueError
        Happens in case if `ignore_missing=False` and there is no
        parameters for some of the layers.
    """
    if load_by not in ('names', 'order', 'names_or_order'):
        raise ValueError(
            "Invalid value for the `load_by` argument: {}. Should be "
            "one of the following values: names, order, names_or_order."
            "".format(load_by))

    if not skip_validation:
        validate_data_structure(data)

    network = extract_network(network)

    # We need to initialize network, to make sure
    # that each layer will generate shared variables
    network.create_variables()

    # We are only interested in layers that has parameters
    layers = data['layers']
    layers_data = [l for l in layers if l['parameters']]
    layers_conn = [l for l in network if l.variables]

    if not ignore_missing and len(layers_data) != len(layers_conn):
        raise ParameterLoaderError(
            "Couldn't load parameters from the dictionary. Connection "
            "has {} layers with parameters whether stored data has {}"
            "".format(len(layers_data), len(layers_conn)))

    if load_by == 'names':
        load_dict_by_names(
            layers_conn, layers_data, ignore_missing)

    elif load_by == 'order':
        load_dict_sequentially(layers_conn, layers_data)

    else:
        try:
            # First we try to load parameters using there names as
            # identifiers. Names are more reliable identifiers than
            # order of layers in the network
            load_dict_by_names(layers_conn, layers_data, ignore_missing)

        except ParameterLoaderError:
            # If we couldn't load data using layer names we will try to
            # compare layers in sequence one by one. Even if names are
            # different networks can be the same and order of parameters
            # should also be the same
            load_dict_sequentially(layers_conn, layers_data)


def save_dict(network):
    """
    Save network into the dictionary.

    Parameters
    ----------
    network : network, list of layer or network

    Returns
    -------
    dict
        Saved parameters and information about network in dictionary
        using specific format. Learn more about the NeuPy's storage
        format in the official documentation.

    Examples
    --------
    >>> from neupy import layers, storage
    >>>
    >>> network = layers.Input(10) >> layers.Softmax(3)
    >>> layers_data = storage.save_dict(network)
    >>>
    >>> layers_data.keys()
    ['layers', 'graph', 'metadata']
    """
    network = extract_network(network)
    network.create_variables()

    session = tf_utils.tensorflow_session()
    tf_utils.initialize_uninitialized_variables()

    data = {
        'metadata': {
            'language': 'python',
            'library': 'neupy',
            'version': neupy.__version__,
            'created': strftime("%a, %d %b %Y %H:%M:%S %Z", gmtime()),
        },
        # Make it as a list in order to save the right order
        # of paramters, otherwise it can be convert to the dictionary.
        'graph': network.layer_names_only(),
        'layers': [],
    }

    for layer in network:
        parameters = {}
        configs = {}

        for attrname, parameter in layer.variables.items():
            parameters[attrname] = {
                'value': asfloat(session.run(parameter)),
                'trainable': parameter.trainable,
            }

        for option_name in layer.options:
            if option_name not in parameters:
                configs[option_name] = getattr(layer, option_name)

        data['layers'].append({
            'class_name': layer.__class__.__name__,
            'name': layer.name,
            'parameters': parameters,
            'configs': configs,
        })

    return data


@shared_docs(save_dict)
def save_pickle(network, filepath, python_compatible=False):
    """
    Save layer parameters in pickle file.

    Parameters
    ----------
    {save_dict.network}

    filepath : str
        Path to the pickle file that stores network parameters.

    python_compatible : bool
        If `True` pickled object would be compatible with
        Python 2 and 3 (pickle protocol equal to `2`).
        If `False` then value would be pickled as highest
        protocol (`pickle.HIGHEST_PROTOCOL`).
        Defaults to `False`.

    Examples
    --------
    >>> from neupy import layers, storage
    >>>
    >>> network = layers.Input(10) > layers.Softmax(3)
    >>> storage.save_pickle(network, '/path/to/parameters.pickle')
    """
    network = extract_network(network)
    data = save_dict(network)

    with open(filepath, 'wb+') as f:
        # Protocol 2 is compatible for both python versions
        protocol = pickle.HIGHEST_PROTOCOL if not python_compatible else 2
        pickle.dump(data, f, protocol)


@shared_docs(load_dict)
def load_pickle(network, filepath, ignore_missing=False,
                load_by='names_or_order', skip_validation=True):
    """
    Load and set parameters for layers from the
    specified filepath.

    Parameters
    ----------
    {load_dict.network}

    filepath : str
        Path to pickle file that will store network parameters.

    {load_dict.ignore_missing}

    {load_dict.load_by}

    {load_dict.skip_validation}

    Raises
    ------
    {load_dict.Raises}

    Examples
    --------
    >>> from neupy import layers, storage
    >>>
    >>> network = layers.Input(10) > layers.Softmax(3)
    >>> storage.load_pickle(network, '/path/to/parameters.pickle')
    """
    network = extract_network(network)

    with open(filepath, 'rb') as f:
        # Specify encoding for python 3 in order to be able to
        # read files that has been created in python 2
        options = {'encoding': 'latin1'} if six.PY3 else {}
        data = pickle.load(f, **options)

    load_dict(network, data, ignore_missing, load_by)


@shared_docs(save_dict)
def save_hdf5(network, filepath):
    """
    Save network parameters in HDF5 format.

    Parameters
    ----------
    {save_dict.network}

    filepath : str
        Path to the HDF5 file that stores network parameters.

    Examples
    --------
    >>> from neupy import layers, storage
    >>>
    >>> network = layers.Input(10) > layers.Softmax(3)
    >>> storage.save_hdf5(network, '/path/to/parameters.hdf5')
    """
    network = extract_network(network)
    data = save_dict(network)

    with h5py.File(filepath, mode='w') as f:
        layer_names = []

        for layer in data['layers']:
            layer_name = layer['name']
            layer_group = f.create_group(layer_name)

            for attrname, attrvalue in layer.items():
                if attrname != 'parameters':
                    layer_group.attrs[attrname] = json.dumps(
                        attrvalue, default=repr)

            for param_name, param in layer['parameters'].items():
                dataset = layer_group.create_dataset(
                    param_name, data=param['value'])

                dataset.attrs['trainable'] = param['trainable']

            layer_names.append(layer_name)

        f.attrs['metadata'] = json.dumps(data['metadata'])
        f.attrs['graph'] = json.dumps(data['graph'])
        f.attrs['layer_names'] = json.dumps(layer_names)


@shared_docs(load_dict)
def load_hdf5(network, filepath, ignore_missing=False,
              load_by='names_or_order', skip_validation=True):
    """
    Load network parameters from HDF5 file.

    Parameters
    ----------
    {load_dict.network}

    filepath : str
        Path to HDF5 file that will store network parameters.

    {load_dict.ignore_missing}

    {load_dict.load_by}

    {load_dict.skip_validation}

    Raises
    ------
    {load_dict.Raises}

    Examples
    --------
    >>> from neupy import layers, storage
    >>>
    >>> network = layers.Input(10) > layers.Softmax(3)
    >>> storage.load_hdf5(network, '/path/to/parameters.hdf5')
    """
    network = extract_network(network)
    data = {}

    with h5py.File(filepath, mode='r') as f:
        data['metadata'] = json.loads(f.attrs['metadata'])
        data['graph'] = json.loads(f.attrs['graph'])
        data['layers'] = []

        layer_names = json.loads(f.attrs['layer_names'])

        for layer_name in layer_names:
            layer_group = f[layer_name]
            layer = {'name': layer_name}

            for attrname, attrvalue in layer_group.attrs.items():
                try:
                    layer[attrname] = json.loads(attrvalue)
                except ValueError:
                    layer[attrname] = attrvalue

            layer['parameters'] = {}
            for param_name, parameter in layer_group.items():
                layer['parameters'][param_name] = {
                    'value': parameter.value,
                    'trainable': parameter.attrs['trainable'],
                }

            data['layers'].append(layer)

    load_dict(network, data, ignore_missing, load_by)


def convert_numpy_array_to_list_recursively(data):
    for key, value in data.items():
        if isinstance(value, dict):
            convert_numpy_array_to_list_recursively(value)

        elif isinstance(value, np.ndarray):
            data[key] = value.tolist()

        elif isinstance(value, list):
            for entity in value:
                if isinstance(entity, dict):
                    convert_numpy_array_to_list_recursively(entity)


@shared_docs(save_dict)
def save_json(network, filepath, indent=None):
    """
    Save network parameters in JSON format.

    Parameters
    ----------
    {save_dict.network}

    filepath : str
        Path to the JSON file that stores network parameters.

    indent : int or None
        Indentation that would be specified for the output JSON.
        Intentation equal to `2` or `4` makes it easy to read raw
        text files. The `None` value disables indentation which means
        that everything will be stored compactly. Defaults to `None`.

    Examples
    --------
    >>> from neupy import layers, storage
    >>>
    >>> network = layers.Input(10) > layers.Softmax(3)
    >>> storage.save_json(network, '/path/to/parameters.json')
    """
    network = extract_network(network)
    data = save_dict(network)

    with open(filepath, 'w') as f:
        # Without extra data processor we won't be able to dump
        # numpy array into json without raising an error.
        # `json` will have issues with numpy array encoding
        convert_numpy_array_to_list_recursively(data)
        return json.dump(data, f, indent=indent, default=repr)


@shared_docs(load_dict)
def load_json(network, filepath, ignore_missing=False,
              load_by='names_or_order', skip_validation=True):
    """
    Load network parameters from JSON file.

    Parameters
    ----------
    {load_dict.network}

    filepath : str
        Path to JSON file that will store network parameters.

    {load_dict.ignore_missing}

    {load_dict.load_by}

    {load_dict.skip_validation}

    Raises
    ------
    {load_dict.Raises}

    Examples
    --------
    >>> from neupy import layers, storage
    >>>
    >>> network = layers.Input(10) > layers.Softmax(3)
    >>> storage.load_json(network, '/path/to/parameters.json')
    """
    network = extract_network(network)
    data = save_dict(network)

    with open(filepath, 'r') as f:
        data = json.load(f)

    load_dict(network, data, ignore_missing, load_by)


# Convenient aliases
save = save_hdf5
load = load_hdf5
