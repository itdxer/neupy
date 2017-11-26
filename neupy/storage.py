import json
import pkgutil
import importlib
from time import gmtime, strftime

import six
import theano
import numpy as np
from six.moves import cPickle as pickle

import neupy
from neupy.utils import asfloat
from neupy.core.docs import shared_docs
from neupy.layers.utils import extract_connection


__all__ = (
    'save', 'load',  # aliases to pickle
    'save_pickle', 'load_pickle',
    'save_json', 'load_json',
    'save_hdf5', 'load_hdf5',
    'load_dict', 'save_dict')


class ParameterLoaderError(Exception):
    """
    Exception triggers in case if there are some issues
    during the parameter loading procedure.
    """


class InvalidFormat(Exception):
    """
    Exception triggers when there are some issue with
    data format that stores connection data.
    """


def load_hdf5_module():
    """
    Loads `h5py` module. THis module used for manipulations
    with HDF5 files.

    Raises
    ------
    ImportError
        In case if module `h5py` wasn't installed

    Returns
    -------
    module
    """
    if not pkgutil.find_loader('h5py'):
        raise ImportError(
            "The `h5py` library wasn't installed. Try to "
            "install it with pip: \n    pip install h5py")

    return importlib.import_module('h5py')


def validate_layer_compatibility(layer, layer_data):
    """
    Checkes if it's possible to load stored data in the
    specified layer.

    Raises
    ------
    ParameterLoaderError
        When there are some problems with stored data.
    """
    expected_input_shape = layer_data['input_shape']
    expected_output_shape = layer_data['output_shape']

    if list(expected_input_shape) != list(layer.input_shape):
        raise ParameterLoaderError(
            "Layer `{}` expected to have input shape equal to {}, got {}"
            "".format(layer.name, expected_input_shape, layer.input_shape))

    if list(expected_output_shape) != list(layer.output_shape):
        raise ParameterLoaderError(
            "Layer `{}` expected to have output shape equal to {}, got {}"
            "".format(layer.name, expected_output_shape,
                      layer.output_shape))


def load_layer_parameter(layer, layer_data):
    """
    Set layer parameters to the values specified in the
    stored data
    """
    for param_name, param_data in layer_data['parameters'].items():
        parameter = getattr(layer, param_name)
        parameter.set_value(asfloat(param_data['value']))


def load_dict_by_names(layers_conn, layers_data, ignore_missed=False):
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

    if not ignore_missed and layers_data.keys() != layers_conn.keys():
        raise ParameterLoaderError(
            "Cannot match layers by name. \n"
            "  Layer names in connection: {}\n"
            "  Layer names in stored data: {}"
            "".format(layers_conn.keys(), layers_data.keys()))

    elif ignore_missed and all(l not in layers_data for l in layers_conn):
        raise ParameterLoaderError("Non of the layers can be matched by name")

    for layer_name, layer in layers_conn.items():
        if layer_name in layers_data:
            validate_layer_compatibility(layer, layers_data[layer_name])

    for layer_name, layer in layers_conn.items():
        if layer_name in layers_data:
            load_layer_parameter(layer, layers_data[layer_name])


def load_dict_sequentially(layers_conn, layers_data):
    """"
    Load parameters in to layer using sequential order of
    layer in connection and stored data
    """
    # It's important to point out that it can be that there more
    # stored layers than specified in the network. For this case we
    # expect to match as much as we can in case if layer are matchable.
    for layer, layer_data in zip(layers_conn, layers_data):
        validate_layer_compatibility(layer, layer_data)

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
                "Layer in the {} position (0-based indeces) is not a "
                "dictionary (it is {})".format(layer_index, type(layer)))

        for attr in ('parameters', 'input_shape', 'output_shape', 'name'):
            if attr not in layer:
                raise InvalidFormat(
                    "Layer in the {} position (0-based indeces) don't "
                    "have key `{}` specified".format(layer_index, attr))

        for attr in ('input_shape', 'output_shape'):
            if not isinstance(layer[attr], (list, tuple)):
                raise InvalidFormat(
                    "{} has invalid format for shape. It should be list "
                    "or tuple, got {}".format(attr, type(layer[attr])))

        if not isinstance(layer['parameters'], dict):
            raise InvalidFormat(
                "Layer in the {} position (0-based indeces) parameters "
                "specified as `{}`, but dictionary expected"
                "".format(layer_index, type(layer['parameters'])))

        for param_name, param in layer['parameters'].items():
            if not isinstance(param, dict):
                raise InvalidFormat(
                    "Layer in the {} position (0-based indeces) has "
                    "incorrect value for parameter named `{}`. It has "
                    "to be a dictionary, but got {}"
                    "".format(layer_index, param_name, type(param)))

            if 'value' not in param:
                raise InvalidFormat(
                    "Layer in the {} position (0-based indeces) has "
                    "incorrect value for parameter named `{}`. Parameter "
                    "doesn't have key named `value`"
                    "".format(layer_index, param_name))


def load_dict(connection, data, ignore_missed=False, load_by='names_or_order'):
    """
    Load network connections from dictionary.

    Parameters
    ----------
    connection : network, list of layer or connection

    data : dict
        Dictionary that stores network parameters.

    ignore_missed : bool
        ``False`` means that error will be triggered in case
        if some of the layers doesn't have storage parameters
        in the specified source. Defaults to ``False``.

    load_by : {{``names``, ``order``, ``names_or_order``}}
        Defines strategy that will be used during parameter loading

        - ``names`` - Matches layers in the network with stored layer
          using their names.

        - ``order`` - Matches layers in the network with stored layer
          using exect order of layers.

        - ``names_or_order`` - Matches layers in the network with stored
          layer trying to do it first using the same names and then
          matching them sequentialy.

        Defaults to ``names_or_order``.

    Raises
    ------
    ValueError
        Happens in case if `ignore_missed=False` and there is no
        parameters for some of the layers.
    """
    if load_by not in ('names', 'order', 'names_or_order'):
        raise ValueError(
            "Invalid value for the `load_by` argument: {}. Should be "
            "one of the following values: names, order, names_or_order."
            "".format(load_by))

    validate_data_structure(data)
    connection = extract_connection(connection)

    # We are only interested in layers that has parameters
    layers = data['layers']
    layers_data = [l for l in layers if l['parameters']]
    layers_conn = [l for l in connection if l.parameters]

    if not ignore_missed and len(layers_data) != len(layers_conn):
        raise ParameterLoaderError(
            "Couldn't load parameters from the dictionary. Connection "
            "has {} layers with parameters whether stored data has {}"
            "".format(len(layers_data), len(layers_conn)))

    if load_by == 'names':
        load_dict_by_names(layers_conn, layers_data, ignore_missed)

    elif load_by == 'order':
        load_dict_sequentially(layers_conn, layers_data)

    else:
        try:
            # First we try to load parameters using there names as
            # identifiers. Names are more reliable identifiers than
            # order of layers in the network
            load_dict_by_names(layers_conn, layers_data, ignore_missed)

        except ParameterLoaderError:
            # If we couldn't load data using layer names we will try to
            # compare layers in sequence one by one. Even if names are
            # different networks can be the same and order of parameters
            # should also be the same
            load_dict_sequentially(layers_conn, layers_data)

    # We need to initalize connection, to make sure
    # that each layer will generate shared variables
    # and validate connections
    connection.initialize()


def save_dict(connection):
    """
    Save network into the dictionary.

    Parameters
    ----------
    connection : network, list of layer or connection

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
    >>> connection = layers.Input(10) > layers.Softmax(3)
    >>> layers_data = storage.save_dict(connection)
    >>>
    >>> layers_data.keys()
    ['layers', 'graph', 'metadata']
    """
    connection = extract_connection(connection)
    data = {
        'metadata': {
            'language': 'python',
            'library': 'neupy',
            'version': neupy.__version__,
            'created': strftime("%a, %d %b %Y %H:%M:%S %Z", gmtime()),
            'theano_float': theano.config.floatX,
        },
        # Make it as a list in order to save the right order
        # of paramters, otherwise it can be convert to the dictionary.
        'graph': connection.graph.layer_names_only(),
        'layers': [],
    }

    for layer in connection:
        parameters = {}
        configs = {}

        for attrname, parameter in layer.parameters.items():
            parameters[attrname] = {
                'value': asfloat(parameter.get_value()),
                'trainable': parameter.trainable,
            }

        for option_name in layer.options:
            if option_name not in parameters:
                configs[option_name] = getattr(layer, option_name)

        data['layers'].append({
            'class_name': layer.__class__.__name__,
            'input_shape': layer.input_shape,
            'output_shape': layer.output_shape,
            'name': layer.name,
            'parameters': parameters,
            'configs': configs,
        })

    return data


@shared_docs(save_dict)
def save_pickle(connection, filepath, python_compatible=True):
    """
    Save layer parameters in pickle file.

    Parameters
    ----------
    {save_dict.connection}

    filepath : str
        Path to the pickle file that stores network parameters.

    python_compatible : bool
        If `True` pickled object would be compatible with
        Python 2 and 3 (pickle protocol equalt to `2`).
        If `False` then value would be pickled as highest
        protocol (`pickle.HIGHEST_PROTOCOL`).
        Defaults to `True`.

    Examples
    --------
    >>> from neupy import layers, storage
    >>>
    >>> connection = layers.Input(10) > layers.Softmax(3)
    >>> storage.save_pickle(connection, '/path/to/parameters.pickle')
    """
    connection = extract_connection(connection)
    data = save_dict(connection)

    with open(filepath, 'wb+') as f:
        # Protocol 2 is compatible for both python versions
        protocol = pickle.HIGHEST_PROTOCOL if python_compatible else 2
        pickle.dump(data, f, protocol)


@shared_docs(load_dict)
def load_pickle(connection, filepath, ignore_missed=False,
                load_by='names_or_order'):
    """
    Load and set parameters for layers from the
    specified filepath.

    Parameters
    ----------
    {load_dict.connection}

    filepath : str
        Path to pickle file that will store network parameters.

    {load_dict.ignore_missed}

    {load_dict.load_by}

    Raises
    ------
    {load_dict.Raises}

    Examples
    --------
    >>> from neupy import layers, storage
    >>>
    >>> connection = layers.Input(10) > layers.Softmax(3)
    >>> storage.load_pickle(connection, '/path/to/parameters.pickle')
    """
    connection = extract_connection(connection)

    with open(filepath, 'rb') as f:
        if six.PY3:
            # Specify encoding for python 3 in order to be able to
            # read files that has been created in python 2
            data = pickle.load(f, encoding='latin1')  # skip coverage
        else:
            data = pickle.load(f)  # skip coverage

    load_dict(connection, data, ignore_missed, load_by)


@shared_docs(save_dict)
def save_hdf5(connection, filepath):
    """
    Save network parameters in HDF5 format.

    Parameters
    ----------
    {save_dict.connection}

    filepath : str
        Path to the HDF5 file that stores network parameters.

    Examples
    --------
    >>> from neupy import layers, storage
    >>>
    >>> connection = layers.Input(10) > layers.Softmax(3)
    >>> storage.save_hdf5(connection, '/path/to/parameters.hdf5')
    """
    hdf5 = load_hdf5_module()
    connection = extract_connection(connection)
    data = save_dict(connection)

    with hdf5.File(filepath, mode='w') as f:
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
def load_hdf5(connection, filepath, ignore_missed=False,
              load_by='names_or_order'):
    """
    Load network parameters from HDF5 file.

    Parameters
    ----------
    {load_dict.connection}

    filepath : str
        Path to HDF5 file that will store network parameters.

    {load_dict.ignore_missed}

    {load_dict.load_by}

    Raises
    ------
    {load_dict.Raises}

    Examples
    --------
    >>> from neupy import layers, storage
    >>>
    >>> connection = layers.Input(10) > layers.Softmax(3)
    >>> storage.load_hdf5(connection, '/path/to/parameters.hdf5')
    """
    hdf5 = load_hdf5_module()
    connection = extract_connection(connection)
    data = {}

    with hdf5.File(filepath, mode='r') as f:
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

    load_dict(connection, data, ignore_missed, load_by)


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


def dump_with_fastest_json_module(data, f, indent=None):
    """
    Data will be dumped using `ujson` module in case if it installed,
    otherwise Python's built-in `json` module will be used.

    Parameters
    ----------
    data : dict

    f : file object
        Opened file that available for writing

    indent : int or None
        JSON indentation.
    """
    # Without extra data processor we won't be able to dump
    # numpy array into json without raising an error.
    # `json` will have issues with numpy array encoding
    # `ujson` will have issue with numpy float or int encoding
    convert_numpy_array_to_list_recursively(data)

    if pkgutil.find_loader('ujson'):
        ujson = importlib.import_module('ujson')

        if indent is None:
            # Indentation functionality behaves differently compare
            # to the one that Pytnon has in built-in json module.
            #
            # Note: indent=0 will give different output
            # for ujson and json
            indent = 0

        return ujson.dump(data, f, indent=indent)
    return json.dump(data, f, indent=indent, default=repr)


def load_with_fastest_json_module(f):
    """
    Data will be loaded using `ujson` module in case if it installed,
    otherwise Python's built-in `json` module will be used.

    Parameters
    ----------
    f : file object
        Opened file that available for reading
    """
    if pkgutil.find_loader('ujson'):
        ujson = importlib.import_module('ujson')
        return ujson.load(f)
    return json.load(f)


@shared_docs(save_dict)
def save_json(connection, filepath, indent=None):
    """
    Save network parameters in JSON format.

    Parameters
    ----------
    {save_dict.connection}

    filepath : str
        Path to the JSON file that stores network parameters.

    indent : int or None
        Indentation that would be specified for the output JSON.
        Intentation equal to `2` or `4` makes it easy to read raw
        text files. The `None` value disables indentation which means
        that everything will be stored compactly. Defaults to `None`.

    Notes
    -----
    Install `ujson` library in order to speed up saving and
    loading procedure.

    Examples
    --------
    >>> from neupy import layers, storage
    >>>
    >>> connection = layers.Input(10) > layers.Softmax(3)
    >>> storage.save_json(connection, '/path/to/parameters.json')
    """
    connection = extract_connection(connection)
    data = save_dict(connection)

    with open(filepath, 'w') as f:
        dump_with_fastest_json_module(data, f)


@shared_docs(load_dict)
def load_json(connection, filepath, ignore_missed=False,
              load_by='names_or_order'):
    """
    Load network parameters from JSON file.

    Parameters
    ----------
    {load_dict.connection}

    filepath : str
        Path to JSON file that will store network parameters.

    {load_dict.ignore_missed}

    {load_dict.load_by}

    Raises
    ------
    {load_dict.Raises}

    Notes
    -----
    Install `ujson` library in order to speed up saving and
    loading procedure.

    Examples
    --------
    >>> from neupy import layers, storage
    >>>
    >>> connection = layers.Input(10) > layers.Softmax(3)
    >>> storage.load_json(connection, '/path/to/parameters.json')
    """
    connection = extract_connection(connection)
    data = save_dict(connection)

    with open(filepath, 'r') as f:
        data = load_with_fastest_json_module(f)

    load_dict(connection, data, ignore_missed, load_by)


# Convenient aliases
save = save_pickle
load = load_pickle
