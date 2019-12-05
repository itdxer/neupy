import os
import sys
import random
from contextlib import contextmanager

import six
import numpy as np
import tensorflow as tf


__all__ = ('as_tuple', 'AttributeKeyDict', 'reproducible', 'extend_error_message_if_fails')


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


@contextmanager
def extend_error_message_if_fails(error_message):
    try:
        yield
    except Exception as exception:
        original_message = str(exception).rstrip('.')
        modified_exception = exception.__class__(original_message + ". " + error_message)

        if hasattr(sys, 'last_traceback') and six.PY3:
            modified_exception = modified_exception.with_traceback(sys.last_traceback)

        raise modified_exception
