import numpy as np
import tensorflow as tf
from scipy.sparse import issparse

from neupy.utils.misc import as_tuple


__all__ = ('format_data', 'asfloat', 'shuffle')


def format_data(data, is_feature1d=True, copy=False, make_float=True):
    """
    Transform data in a standardized format.

    Notes
    -----
    It should be applied to the input data prior to use in
    learning algorithms.

    Parameters
    ----------
    data : array-like
        Data that should be formated. That could be, matrix, vector or
        Pandas DataFrame instance.

    is_feature1d : bool
        Should be equal to ``True`` if input data is a vector that
        contains N samples with 1 feature each. Defaults to ``True``.

    copy : bool
        Defaults to ``False``.

    make_float : bool
        If `True` then input will be converted to float.
        Defaults to ``False``.

    Returns
    -------
    ndarray
        The same input data but transformed to a standardized format
        for further use.
    """
    if data is None or issparse(data):
        return data

    if make_float:
        data = asfloat(data)

    if not isinstance(data, np.ndarray) or copy:
        data = np.array(data, copy=copy)

    # Valid number of features for one or two dimensions
    n_features = data.shape[-1]

    if data.ndim == 1:
        data_shape = (n_features, 1) if is_feature1d else (1, n_features)
        data = data.reshape(data_shape)

    return data


def asfloat(value):
    """
    Convert variable to 32 bit float number.

    Parameters
    ----------
    value : matrix, ndarray, Tensorfow variable or scalar
        Value that could be converted to float type.

    Returns
    -------
    matrix, ndarray, Tensorfow variable or scalar
        Output would be input value converted to 32 bit float.
    """
    float_type = 'float32'

    if isinstance(value, (np.matrix, np.ndarray)):
        if value.dtype != np.dtype(float_type):
            return value.astype(float_type)

        return value

    elif isinstance(value, (tf.Tensor, tf.SparseTensor)):
        return tf.cast(value, tf.float32)

    elif issparse(value):
        return value

    float_x_type = np.cast[float_type]
    return float_x_type(value)


def shuffle(X, y):
    """
    Randomly shuffle rows in the arrays without breaking
    associations between rows in ``X`` and ``y``.

    Parameters
    ----------
    X : array-line
    y : array-line

    Returns
    -------
    tupe
        Shuffled ``X`` and ``y``.
    """
    arrays = as_tuple(X, y)
    filtered_arrays = tuple(array for array in arrays if array is not None)

    if not filtered_arrays:
        return arrays

    first = filtered_arrays[0]
    n_samples = first.shape[0]

    if any(n_samples != array.shape[0] for array in filtered_arrays):
        array_shapes = [array.shape for array in filtered_arrays]
        raise ValueError("Cannot shuffle matrices. All matrices should "
                         "have the same number of rows. Input shapes are: {}"
                         "".format(array_shapes))

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    arrays = list(arrays)
    for i, array in enumerate(arrays):
        if array is not None:
            arrays[i] = array[indices]

    X = arrays[:-1] if len(arrays) > 2 else arrays[0]
    y = arrays[-1]

    return X, y
