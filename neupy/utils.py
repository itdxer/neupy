import sys

import numpy as np


__all__ = ('format_data', 'is_row1d')


def format_data(input_data, row1d=False, copy=False):
    if input_data is None:
        return

    input_data = np.array(input_data, copy=copy)

    # Valid number of features for one or two dimentions
    n_features = input_data.shape[-1]
    if 'pandas' in sys.modules:
        pandas = sys.modules['pandas']

        if isinstance(input_data, (pandas.Series, pandas.DataFrame)):
            input_data = input_data.values
    if input_data.ndim == 1:
        data_shape = (1, n_features) if row1d else (n_features, 1)
        input_data = input_data.reshape(data_shape)

    return input_data


def is_row1d(layer):
    if layer is None:
        return False
    return (layer.input_size != 1)
