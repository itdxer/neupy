import theano.tensor as T


__all__ = ('preformat_layer_shape', 'dimshuffle', 'iter_parameters',
           'count_parameters', 'create_input_variable')


def preformat_layer_shape(shape):
    """
    Format layer's input or output shape.

    Parameters
    ----------
    shape : int or tuple

    Returns
    -------
    int or tuple
    """
    if isinstance(shape, tuple) and len(shape) == 1:
        return shape[0]
    return shape


def dimshuffle(value, ndim, axes):
    """
    Shuffle dimension based on the specified number of
    dimensions and axes.

    Parameters
    ----------
    value : Theano variable
    ndim : int
    axes : tuple, list

    Returns
    -------
    Theano variable
    """
    pattern = ['x'] * ndim

    for i, axis in enumerate(axes):
        pattern[axis] = i

    return value.dimshuffle(pattern)


def iter_parameters(layers, only_trainable=True):
    """
    Iterate through layer parameters.

    Parameters
    ----------
    layers : list of layers or connection

    Yields
    ------
    tuple
        Tuple with three ariables: (layer, attribute_name, parameter)
    """
    observed_parameters = []
    for layer in layers:
        for attrname, parameter in layer.parameters.items():
            new_parameter = parameter not in observed_parameters
            if new_parameter and (parameter.trainable or not only_trainable):
                observed_parameters.append(parameter)
                yield layer, attrname, parameter


def count_parameters(connection):
    """
    Count number of parameters in Neural Network.

    Parameters
    ----------
    connection : list of laters or connection

    Returns
    -------
    int
        Number of parameters.
    """
    n_parameters = 0

    for _, _, parameter in iter_parameters(connection):
        parameter = parameter.get_value()
        n_parameters += parameter.size

    return n_parameters


def create_input_variable(input_shape, name):
    """
    Create input variable based on the specified
    input shape.

    Parameters
    ----------
    input_shape : tuple
    name : str

    Returns
    -------
    Theano variable
    """
    dim_to_variable_type = {
        2: T.matrix,
        3: T.tensor3,
        4: T.tensor4,
    }

    # Shape doesn't include batch size dimension,
    # that's why we need to add one
    ndim = len(input_shape) + 1

    if ndim not in dim_to_variable_type:
        raise ValueError("Layer's input needs to be 2, 3 or 4 "
                         "dimensional. Found {} dimensions".format(ndim))

    variable_type = dim_to_variable_type[ndim]
    return variable_type(name)
