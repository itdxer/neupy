import os
import random

import tensorflow as tf
import numpy as np


__all__ = (
    'as_tuple', 'number_type', 'AttributeKeyDict',
    'all_equal', 'reproducible',
)


number_type = (int, float, np.floating, np.integer)


def as_tuple(*values):
    """
    Convert sequence of values in one big tuple.

    Parameters
    ----------
    *values
        Values that needs to be combined in one big tuple.

    Returns
    -------
    tuple
        All input values combined in one tuple

    Examples
    --------
    >>> as_tuple(None, (1, 2, 3), None)
    (None, 1, 2, 3, None)
    >>>
    >>> as_tuple((1, 2, 3), (4, 5, 6))
    (1, 2, 3, 4, 5, 6)
    """
    cleaned_values = []
    for value in values:
        if isinstance(value, (tuple, list)):
            cleaned_values.extend(value)
        else:
            cleaned_values.append(value)
    return tuple(cleaned_values)


class AttributeKeyDict(dict):
    """
    Modified built-in Python ``dict`` class. That modification
    helps get and set values like attributes.

    Examples
    --------
    >>> attrdict = AttributeKeyDict()
    >>> attrdict
    {}
    >>> attrdict.test_key = 'test_value'
    >>> attrdict
    {'test_key': 'test_value'}
    >>> attrdict.test_key
    'test_value'
    """
    def __getattr__(self, attrname):
        return self[attrname]

    def __setattr__(self, attrname, value):
        self[attrname] = value

    def __delattr__(self, attrname):
        del self[attrname]

    def __reduce__(self):
        return (self.__class__, (dict(self),))


def all_equal(array):
    """
    Checks if all elements in the array are equal.

    Parameters
    ----------
    array : list, tuple

    Raises
    ------
    ValueError
        If input array is empty

    Returns
    -------
    bool
        `True` in case if all elements are equal and
        `False` otherwise.
    """
    if not array:
        raise ValueError("Array is empty")

    first_item = array[0]

    if any(item != first_item for item in array):
        return False

    return True


def reproducible(seed=0):
    """
    Set up the same seed value for the NumPy and
    python random module to make your code reproducible.

    Parameters
    ----------
    seed : int
        Defaults to ``0``.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
