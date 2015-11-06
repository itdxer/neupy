import os
import copy
import tempfile
from contextlib import contextmanager

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images


def compare_networks(default_class, tested_class, data, **kwargs):
    epochs = kwargs.pop('epochs', 100)
    is_comparison_plot = kwargs.pop('is_comparison_plot', False)

    # Compute result for default network (which must be slower)
    network = default_class(**kwargs)

    default_connections = copy.deepcopy(network.connection)
    network.train(*data, epochs=epochs)

    network_default_error = network.last_error()
    errors1 = network.errors_in

    # Compute result for test network (which must be faster)
    kwargs['connection'] = default_connections
    network = tested_class(**kwargs)

    network.train(*data, epochs=epochs)
    network_tested_error = network.last_error()
    errors2 = network.errors_out

    if is_comparison_plot:
        error_range = np.arange(max(len(errors1), len(errors2)))
        plt.plot(error_range[:len(errors1)], errors1)
        plt.plot(error_range[:len(errors2)], errors2)
        plt.show()

    return network_default_error, network_tested_error


@contextmanager
def image_comparison(original_image_path, figsize=(10, 10), tol=1e-3):
    currentdir = os.path.abspath(os.path.dirname(__file__))
    original_image_path = os.path.join(currentdir, original_image_path)

    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        figure = plt.figure(figsize=figsize)
        yield figure
        figure.savefig(f.name)
        error = compare_images(f.name, original_image_path, tol=tol)

        if error:
            raise AssertionError("Image comparison failed. \n"
                                 "Information: {}".format(error))
