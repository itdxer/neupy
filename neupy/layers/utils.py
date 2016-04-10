import importlib

import numpy as np

from neupy.layers.connections import LayerConnection


__all__ = ('generate_layers', 'random_orthogonal', 'random_bounded',
           'generate_weight')


DEFAULT_LAYER_CLASS = "neupy.layers.Sigmoid"
DEFAULT_OUTPUT_LAYER_CLASS = "neupy.layers.Output"


def import_class(object_path):
    """ Import class from module using module path written as Python
    string.

    Parameters
    ----------
    object_path : str
        Path to the object. For example, it can be written
        as ``'path.to.module.MyClass'``.

    Returns
    -------
    object
        Function returns object imported using identified path.
    """

    module_name, classname = object_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, classname)


def generate_layers(layers_sizes):
    """ Create from list of layer sizes basic linear network.

    Parameters
    ----------
    layers_sizes : list or tuple
        Ordered list of network connection structure.

    Returns
    -------
    LayerConnection
        Constructed connection.
    """

    if len(layers_sizes) < 2:
        raise ValueError("Network must contains at least 2 layers.")

    default_layer_class = import_class(DEFAULT_LAYER_CLASS)
    default_output_layer_class = import_class(DEFAULT_OUTPUT_LAYER_CLASS)

    output_layer_size = layers_sizes.pop()
    connection = default_output_layer_class(output_layer_size)

    for input_size in reversed(layers_sizes):
        left_layer = default_layer_class(input_size)
        connection = LayerConnection(left_layer, connection)

    return connection


def random_orthogonal(shape):
    """ Build random orthogonal 2D matrix.

    Parameters
    ----------
    shape : tuple
        Generated matrix shape.

    Returns
    -------
    ndarray
        Orthogonalized random matrix.
    """

    if len(shape) not in (1, 2):
        raise ValueError("Shape attribute should be a tuple which contains "
                         "1 or 2 integer values.")

    rand_matrix = np.random.randn(*shape)

    if len(shape) == 1:
        return rand_matrix

    nrows, ncols = shape
    u, _, v = np.linalg.svd(rand_matrix, full_matrices=False)
    ortho_base = u if nrows > ncols else v

    return ortho_base[:nrows, :ncols]


def random_bounded(shape, bounds=(-0.01, 0.01)):
    """ Generate uniform random matrix which values between bounds.

    Parameters
    ----------
    shape : tuple
        Generated matrix shape.
    bounds : list or tuple
        Set up generated weights between bounds. Values should be tuple or
        list that contains two numbers. Defaults to ``(0, 1)``.

    Returns
    -------
    ndarray
        Random matrix between selected bounds.
    """

    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
        raise ValueError("`{}` is invalid value for `bounds` parameter. "
                         "Value should be a tuple or a list that contains "
                         "two numbers.".format(bounds))

    left_bound, right_bound = bounds
    random_weight = np.random.random(shape)
    return random_weight * (right_bound - left_bound) + left_bound


def identify_fans(shape):
    """ Identify fans from shape.

    Parameters
    ----------
    shape : tuple or list
        Matrix shape.

    Returns
    -------
    tuple
        Tuple that contains :math:`fan_{in}` and :math:`fan_{out}`.
    """
    fan_in = shape[0]
    output_feature_shape = shape[1:]

    if output_feature_shape:
        fan_out = np.prod(output_feature_shape).item(0)
    else:
        fan_out = 1

    return fan_in, fan_out


def he_normal(shape):
    """ Kaiming He weight initialization from normal
    distribution.

    Parameters
    ----------
    shape : tuple or list
        Random matrix shape.

    Returns
    -------
    array-like
        Randomly generate matrix with defined shape.

    ..[1] Kaiming He, Xiangyu Zhan, Shaoqing Ren, Jian Sun.
        Delving Deep into Rectifiers: Surpassing Human-Level Performance
        on ImageNet Classification, 2015.
    """
    fan_in, _ = identify_fans(shape)
    variance = 2. / fan_in
    std = np.sqrt(variance)
    return np.random.normal(loc=0, scale=std, size=shape)


def he_uniform(shape):
    """ Kaiming He weight initialization from uniform
    distribution.

    Parameters
    ----------
    shape : tuple or list
        Random matrix shape.

    Returns
    -------
    array-like
        Randomly generate matrix with defined shape.

    ..[1] Kaiming He, Xiangyu Zhan, Shaoqing Ren, Jian Sun.
        Delving Deep into Rectifiers: Surpassing Human-Level Performance
        on ImageNet Classification, 2015.
    """
    fan_in, _ = identify_fans(shape)
    variance = 6. / fan_in
    abs_max_value = np.sqrt(variance)
    return random_bounded(shape, bounds=(-abs_max_value, abs_max_value))


def xavier_normal(shape):
    """ Xavier Glorot weight initialization from normal
    distribution.

    Parameters
    ----------
    shape : tuple or list
        Random matrix shape.

    Returns
    -------
    array-like
        Randomly generate matrix with defined shape.

    ..[1] X Glorot, Y Bengio. Understanding the difficulty of training
        deep feedforward neural networks, 2010.
    """
    fan_in, fan_out = identify_fans(shape)
    variance = 2. / (fan_in + fan_out)
    std = np.sqrt(variance)
    return np.random.normal(loc=0, scale=std, size=shape)


def xavier_uniform(shape):
    """ Xavier Glorot weight initialization from uniform
    distribution.

    Parameters
    ----------
    shape : tuple or list
        Random matrix shape.

    Returns
    -------
    array-like
        Randomly generate matrix with defined shape.

    ..[1] X Glorot, Y Bengio. Understanding the difficulty of training
        deep feedforward neural networks, 2010.
    """
    fan_in, fan_out = identify_fans(shape)
    variance = 6. / (fan_in + fan_out)
    abs_max_value = np.sqrt(variance)
    return random_bounded(shape, bounds=(-abs_max_value, abs_max_value))


NORMAL = 'normal'
BOUNDED = 'bounded'
ORTHOGONAL = 'ortho'
XAVIER_NORMAL = 'xavier_normal'
XAVIER_UNIFORM = 'xavier_uniform'
HE_NORMAL = 'he_normal'
HE_UNIFORM = 'he_uniform'

VALID_INIT_METHODS = (NORMAL, BOUNDED, ORTHOGONAL, XAVIER_NORMAL,
                      XAVIER_UNIFORM, HE_NORMAL, HE_UNIFORM)


def generate_weight(shape, bounds=None, init_method=XAVIER_NORMAL):
    """ Generate random weights for neural network connections.

    Parameters
    ----------
    shape : tuple of int
        Weight shape.
    bounds : tuple of int
        Available only for ``init_method`` equal to ``bounded``.
        Value identify minimum and maximum possible value in random weights.
    init_method : {'bounded', 'normal', 'ortho', 'xavier_normal',\
    'xavier_uniform', 'he_normal', 'he_uniform'}
        Weight initialization method. Defaults to ``xavier_normal``.

        * ``normal`` will generate random weights from normal distribution \
        with standard deviation equal to ``0.01``.

        * ``bounded`` generate random weights from Uniform distribution.

        * ``ortho`` generate random orthogonal matrix.

        * ``xavier_normal`` generate random matrix from normal distrubtion \
        where variance equal to :math:`\\frac{2}{fan_{in} + fan_{out}}`. \
        Where :math:`fan_{in}` is a number of layer input units and \
        :math:`fan_{out}` - number of layer output units.

        * ``xavier_uniform`` generate random matrix from uniform \
        distribution where :math:`w_{ij} \in [-\\sqrt{\\frac{6}{fan_{in} \
        + fan_{out}}}, \\sqrt{\\frac{6}{fan_{in} + fan_{out}}}]`.

        * ``he_normal`` generate random matrix from normal distrubtion \
        where variance equal to :math:`\\frac{2}{fan_{in}}`. \
        Where :math:`fan_{in}` is a number of layer input units.

        * ``he_uniform`` generate random matrix from uniformal \
        distribution where :math:`w_{ij} \in [-\\sqrt{\\frac{6}{fan_{in}}}, \
        \\sqrt{\\frac{6}{fan_{in}}}]`

    Returns
    -------
    ndarray
        Random weight.
    """

    if init_method not in VALID_INIT_METHODS:
        raise ValueError("Undefined initialization method `{}`. Available: {}"
                         "".format(init_method, VALID_INIT_METHODS))

    methods_without_parameters = {
        ORTHOGONAL: random_orthogonal,
        XAVIER_NORMAL: xavier_normal,
        XAVIER_UNIFORM: xavier_uniform,
        HE_NORMAL: he_normal,
        HE_UNIFORM: he_uniform,
    }

    if init_method == NORMAL:
        # TODO: Scale parameter needs to be specified by user.
        weight = np.random.normal(loc=0, scale=0.01, size=shape)

    elif init_method == BOUNDED:
        weight = random_bounded(shape, bounds)

    else:
        generation_function = methods_without_parameters[init_method]
        weight = generation_function(shape)

    return weight
