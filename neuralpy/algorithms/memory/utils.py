__all__ = ('sign2bin', 'bin2sign')


def sign2bin(matrix):
    return (matrix + 1) / 2


def bin2sign(matrix):
    return 2 * matrix - 1
