__all__ = ('sign2bin', 'bin2sign', 'hopfield_energy', 'format_data')


def sign2bin(matrix):
    return (matrix + 1) / 2


def bin2sign(matrix):
    return 2 * matrix - 1


def hopfield_energy(weight, input_data, output_data):
    energy_output = -0.5 * input_data.dot(weight).dot(output_data.T)
    return energy_output.item(0)


def format_data(input_data):
    # Valid number of features for one or two dimentions
    n_features = input_data.shape[-1]
    if input_data.ndim == 1:
        input_data = input_data.reshape((1, n_features))
    return input_data
