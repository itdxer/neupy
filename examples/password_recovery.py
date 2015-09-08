import math
import random
import string
import binascii
from itertools import repeat
from collections import OrderedDict
from operator import itemgetter

import numpy as np
from neupy import algorithms


def str2bin(text, max_length=30, encoding='ascii'):
    if len(text) > max_length:
        raise ValueError("Text can't contains more "
                         "than {} symbols".format(max_length))

    text = text.rjust(max_length)
    byte_text = bytearray(text, encoding)
    binary_string = bin(int(binascii.hexlify(byte_text), 16))

    # Remove 0b symbols from string
    binary_array = map(int, binary_string[2:])

    # Python cut leading zeros from the beggining,
    # so we need put them back
    letter_bin_len = math.ceil((len(binary_string) - 2) / max_length)
    valid_length = letter_bin_len * max_length
    bin_vector = list(binary_array)

    n_leading_zeros = valid_length - len(bin_vector)
    leading_zeros = list(repeat(0, times=n_leading_zeros))

    return leading_zeros + bin_vector


def bin2str(array, encoding='ascii'):
    string_array = map(str, array)
    binary_string = ''.join(string_array)
    binary_string = '0b' + binary_string
    hex_string = bytearray(hex(int(binary_string, 2)), encoding)
    raw_text = binascii.unhexlify(hex_string[2:])
    return raw_text.lstrip().decode(encoding)


def generate_password(min_length=5, max_length=30):
    symbols = list(
        string.ascii_letters +
        string.digits +
        string.punctuation
    )
    password_len = random.randint(min_length, max_length + 1)
    password = [np.random.choice(symbols) for _ in range(password_len)]
    return ''.join(password)


def save_password(real_password, noize_level=5):
    if noize_level < 1:
        raise ValueError("`noize_level` must be equal or greater than 1.")

    binary_password = str2bin(real_password)
    bin_password_len = len(binary_password)

    data = [binary_password]

    for _ in range(noize_level):
        # The farther from the 0.5 value the less likely
        # password recovery
        noize = np.random.binomial(1, 0.55, bin_password_len)
        data.append(noize)

    dhnet = algorithms.DiscreteHopfieldNetwork(mode='full')
    dhnet.train(np.array(data))

    return dhnet


def recover_password(dhnet, broken_password):
    test = np.array(str2bin(broken_password))
    recovered_password = dhnet.predict(test)

    try:
        if recovered_password.ndim == 2:
            recovered_password = recovered_password[0, :]
        password = bin2str(recovered_password)

    except (UnicodeDecodeError, binascii.Error):
        # Panic mode
        password = generate_password()

    return password


def cutword(word, k, fromleft=False):
    if fromleft:
        return (word[-k:] if k != 0 else '').rjust(len(word))
    return (word[:k] if k != 0 else '').ljust(len(word))


def cripple_password(word, k):
    crippled_password = random.sample(list(enumerate(word)), k=k)
    word_letters_list = map(itemgetter(1), sorted(crippled_password))
    return ''.join(word_letters_list)


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    n_times = 10000
    cases = OrderedDict([
        ('exclude-one', (lambda x: x - 1)),
        ('exclude-quarter', (lambda x: 3 * x // 4)),
        ('exclude-half', (lambda x: x // 2)),
        ('just-one-symbol', (lambda x: 1)),
        ('empty-string', (lambda x: 0)),
    ])
    results = OrderedDict.fromkeys(cases.keys(), 0)

    for _ in range(n_times):
        real_password = generate_password(min_length=25, max_length=25)

        for casename, func in cases.items():
            n_letters = func(len(real_password))
            broken_password = cutword(real_password, k=n_letters,
                                      fromleft=True)

            dhnet = save_password(real_password, noize_level=11)
            recovered_password = recover_password(dhnet, broken_password)

            if recovered_password != real_password:
                results[casename] += 1

    print("Number of fails for each test case:")
    pprint.pprint(results)
