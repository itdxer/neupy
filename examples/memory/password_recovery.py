from __future__ import division, print_function

import random
import pprint
import string
from collections import OrderedDict
from operator import itemgetter

import numpy as np
from tqdm import tqdm
from neupy import algorithms


def str2bin(text, max_length=30):
    if len(text) > max_length:
        raise ValueError("Text can't contains more "
                         "than {} symbols".format(max_length))

    text = text.rjust(max_length)

    bits_list = []
    for symbol in text:
        bits = bin(ord(symbol))
        # Cut `0b` from the beggining and fill with zeros if they
        # are missed
        bits = bits[2:].zfill(8)
        bits_list.extend(map(int, bits))

    return list(bits_list)


def chunker(sequence, size):
    for position in range(0, len(sequence), size):
        yield sequence[position:position + size]


def bin2str(array):
    characters = []
    for binary_symbol_code in chunker(array, size=8):
        binary_symbol_str = ''.join(map(str, binary_symbol_code))
        character = chr(int(binary_symbol_str, base=2))
        characters.append(character)
    return ''.join(characters).lstrip()


def generate_password(min_length=5, max_length=30):
    symbols = list(
        string.ascii_letters +
        string.digits +
        string.punctuation
    )
    password_len = random.randrange(min_length, max_length + 1)
    password = [np.random.choice(symbols) for _ in range(password_len)]
    return ''.join(password)


def save_password(real_password, noise_level=5):
    if noise_level < 1:
        raise ValueError("`noise_level` must be equal or greater than one")

    binary_password = str2bin(real_password)
    bin_password_len = len(binary_password)

    data = [binary_password]

    for _ in range(noise_level):
        # The farther from the 0.5 value the less likely
        # password recovery
        noise = np.random.binomial(1, 0.55, bin_password_len)
        data.append(noise)

    dhnet = algorithms.DiscreteHopfieldNetwork(mode='sync')
    dhnet.train(np.array(data))

    return dhnet


def recover_password(dhnet, broken_password):
    test = np.array(str2bin(broken_password))
    recovered_password = dhnet.predict(test)

    if recovered_password.ndim == 2:
        recovered_password = recovered_password[0, :]

    return bin2str(recovered_password)


def cutword(word, k, fromleft=False):
    if fromleft:
        return (word[-k:] if k != 0 else '').rjust(len(word))
    return (word[:k] if k != 0 else '').ljust(len(word))


def cripple_password(word, k):
    crippled_password = random.sample(list(enumerate(word)), k=k)
    word_letters_list = map(itemgetter(1), sorted(crippled_password))
    return ''.join(word_letters_list)


def loss_of_chars(word, k):
    word_with_missed_chars = cripple_password(word, k)
    broken_word = []

    for symbol in word:
        if word_with_missed_chars.startswith(symbol):
            word_with_missed_chars = word_with_missed_chars[1:]
            broken_word.append(symbol)
        else:
            broken_word.append(' ')

    return ''.join(broken_word)


if __name__ == '__main__':
    n_times = 10000
    cases = OrderedDict([
        ('exclude-one', (lambda x: x - 1)),
        ('exclude-quarter', (lambda x: 3 * x // 4)),
        ('exclude-half', (lambda x: x // 2)),
        ('just-one-symbol', (lambda x: 1)),
        ('empty-string', (lambda x: 0)),
    ])
    results = OrderedDict.fromkeys(cases.keys(), 0)

    for _ in tqdm(range(n_times)):
        real_password = generate_password(min_length=25, max_length=25)

        for casename, func in cases.items():
            n_letters = func(len(real_password))
            broken_password = cutword(real_password, k=n_letters,
                                      fromleft=True)

            dhnet = save_password(real_password, noise_level=11)
            recovered_password = recover_password(dhnet, broken_password)

            if recovered_password != real_password:
                results[casename] += 1

    print("Number of fails for each test case:")
    pprint.pprint(results)
