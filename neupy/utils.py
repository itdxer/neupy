__all__ = ('format_data',)


def format_data(input_data):
    # Valid number of features for one or two dimentions
    n_features = input_data.shape[-1]
    if input_data.ndim == 1:
        input_data = input_data.reshape((1, n_features))
    return input_data
