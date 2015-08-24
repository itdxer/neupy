import inspect
import importlib

import numpy as np


__all__ = ('preformat_value', 'import_class')


def preformat_value(value):
    if inspect.isfunction(value) or inspect.isclass(value):
        return value.__name__

    elif isinstance(value, (list, tuple, set)):
        return [preformat_value(v) for v in value]

    elif isinstance(value, (np.ndarray, np.matrix)):
        return value.shape

    return value


def import_class(path_to_class):
    module_name, classname = path_to_class.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, classname)
