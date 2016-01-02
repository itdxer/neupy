from itertools import chain

import theano.tensor as T


def iter_parameters(network):
    """ Iterate over all network parameters.

    Parameters
    ----------
    network : ConstructableNetwork instance

    Returns
    -------
    iterator
        Returns iterator that contains all weights and biases from the
        network. Parameters from the first layer will be at the beggining
        and the other will be in the same order as layers in the
        network.
    """
    parameters = [layer.parameters for layer in network.train_layers]
    return chain(*parameters)


def parameters2vector(network):
    """ Concatenate all network parameters in one big vector.

    Parameters
    ----------
    network : ConstructableNetwork instance

    Returns
    -------
    object
        Returns concatenated parameters in one big vector.
    """
    params = iter_parameters(network)
    return T.concatenate([param.flatten() for param in params])
