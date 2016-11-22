from itertools import chain

import theano.tensor as T


__all__ = ('parameters2vector', 'iter_parameter_values',
           'setup_parameter_updates')


def iter_parameter_values(network):
    """
    Iterate over all network parameters.

    Parameters
    ----------
    network : ConstructableNetwork instance

    Returns
    -------
    iterator
        Returns iterator that contains all weights and biases
        from the network. Parameters from the first layer will
        be at the beggining and the other will be in the same
        order as layers in the network.
    """
    parameters = [layer.parameters.values() for layer in network.layers]
    return chain(*parameters)


def parameters2vector(network):
    """
    Concatenate all network parameters in one big vector.

    Parameters
    ----------
    network : ConstructableNetwork instance

    Returns
    -------
    object
        Returns all parameters concatenated in one big vector.
    """
    params = iter_parameter_values(network)
    return T.concatenate([param.flatten() for param in params])


def setup_parameter_updates(parameters, parameter_update_vector):
    """
    Creates update rules for list of parameters from one vector.
    Function is useful in Conjugate Gradient or
    Levenberg-Marquardt optimization algorithms

    Parameters
    ----------
    parameters : list
        List of parameters.

    parameter_update_vector : Theano varible
        Vector that contains updates for all parameters.

    Returns
    -------
    list
        List of updates separeted for each parameter.
    """
    updates = []
    start_position = 0

    for parameter in parameters:
        end_position = start_position + parameter.size

        new_parameter = T.reshape(
            parameter_update_vector[start_position:end_position],
            parameter.shape
        )
        updates.append((parameter, new_parameter))

        start_position = end_position

    return updates
