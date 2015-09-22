from numpy import where
from numpy.core.umath_tests import inner1d


__all__ = ('sign2bin', 'bin2sign', 'hopfield_energy')


def sign2bin(matrix):
    return where(matrix == 1, 1, 0)


def bin2sign(matrix):
    return where(matrix == 0, -1, 1)


def hopfield_energy(weight, input_data, output_data):
    return -0.5 * inner1d(input_data.dot(weight), output_data)
