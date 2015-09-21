from numpy import where


__all__ = ('sign2bin', 'bin2sign', 'hopfield_energy')


def sign2bin(matrix):
    return where(matrix == 1, 1, 0)


def bin2sign(matrix):
    return where(matrix == 0, -1, 1)


def hopfield_energy(weight, input_data, output_data):
    energy_output = -0.5 * input_data.dot(weight).dot(output_data.T)
    return energy_output.item(0)
