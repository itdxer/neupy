import copy

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt


def remove_last_zeros(array):
    while len(array) and array[-1] == 0:
        array.pop()
    return array


def errors_stable_at(array):
    return len(remove_last_zeros(array))


def compare_networks(default_class, tested_class, data, **kwargs):
    epochs = kwargs.pop('epochs', 100)
    is_comparison_plot = kwargs.pop('is_comparison_plot', False)

    # Compute result for default network (which must be slower)
    network = default_class(**kwargs)

    default_connections = copy.deepcopy(network.connection)
    network.train(*data, epochs=epochs)

    network_default_error = network.last_error_in()
    errors1 = network.normalized_errors_in()

    # Compute result for test network (which must be faster)
    kwargs['connection'] = default_connections
    network = tested_class(**kwargs)

    network.train(*data, epochs=epochs)
    network_tested_error = network.last_error_in()
    errors2 = network.normalized_errors_in()

    if is_comparison_plot:
        error_range = np.arange(max(len(errors1), len(errors2)))
        plt.plot(error_range[:len(errors1)], errors1)
        plt.plot(error_range[:len(errors2)], errors2)
        plt.show()

    return network_default_error, network_tested_error


def is_array_or_matrix(value):
    return isinstance(value, (np.ndarray, np.matrix))
