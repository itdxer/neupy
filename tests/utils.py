import os
import copy
import tempfile
from contextlib import contextmanager

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images

from neupy import algorithms, layers

from data import xor_input_train, xor_target_train


def compare_networks(default_class, tested_class, data, **kwargs):
    """ Compare two network arcitectures.

    Parameters
    ----------
    default_class : BaseNetwork instance
    tested_class : BaseNetwork instance
    data : tuple
    **kwargs :

    Raises
    ------
    AssertionError
        Raise exception when first network have better prediction
        accuracy then the second one.
    """
    epochs = kwargs.pop('epochs', 100)
    show_comparison_plot = kwargs.pop('show_comparison_plot', False)

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
    errors2 = network.errors_in

    if show_comparison_plot:
        error_range = np.arange(max(len(errors1), len(errors2)))
        plt.plot(error_range[:len(errors1)], errors1)
        plt.plot(error_range[:len(errors2)], errors2)
        plt.show()

    if network_default_error <= network_tested_error:
        raise AssertionError("First network has smaller error ({}) that the "
                             "second one ({}).".format(network_default_error,
                                                       network_tested_error))


@contextmanager
def image_comparison(original_image_path, figsize=(10, 10), tol=1e-3):
    """ Context manager that initialize figure that should contain figure
    that should be compared with expected one.

    Parameters
    ----------
    original_image_path : str
        Path to original image that will use for comparison.
    figsize : tuple
        Figure size. Defaults to ``(10, 10)``.
    tol : float
        Comparison tolerance. Defaults to ``1e-3``.

    Raises
    ------
    AssertionError
        Exception would be trigger in case when generated images and
        original one are different.
    """
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


def reproducible_network_train(seed=0, epochs=500, **additional_params):
    """ Make a reproducible train for Gradient Descent based neural
    network with a XOR problem and return trained network.

    Parameters
    ----------
    seed : int
        Random State seed number for reproducibility. Defaults to ``0``.
    epochs : int
        Number of epochs for training. Defaults to ``500``.
    **additional_params
        Aditional parameters for Neural Network.

    Returns
    -------
    GradientDescent instance
        Returns trained network.
    """
    np.random.seed(seed)
    network = algorithms.GradientDescent(
        connection=[
            layers.Tanh(2),
            layers.Tanh(5),
            layers.StepOutput(1, output_bounds=(-1, 1))
        ],
        **additional_params
    )
    network.train(xor_input_train, xor_target_train, epochs=epochs)
    return network


def vectors_for_testing(vector, is_feature1d=True):
    """ Function generate different possible variations of one vector.
    That feature useful for testing algorithms input data.

    Parameters
    ----------
    vector : ndarray
        Vector that would be transformed in different data types.
    is_feature1d : bool
        Parameter explain the vector type. Parameter equal to ``True`` mean
        that input data a banch of samples that contains one feature each.
        Defaults to ``True``.

    Raises
    ------
    ValueError
        If input is not a vector

    Returns
    -------
    list
        List that contains the same vectors in different data types like
        numpy 2D vector or pandas Data Frame
    """

    if vector.ndim != 1 and min(vector.shape) != 1:
        raise ValueError("Input should be a vector")

    shape2d = (vector.size, 1) if is_feature1d else (1, vector.size)

    vectors_list = []
    if vector.ndim == 1:
        vectors_list.extend([vector, pd.Series(vector)])

    vectors_list.extend([vector.reshape(shape2d),
                         pd.DataFrame(vector.reshape(shape2d))])

    return vectors_list
