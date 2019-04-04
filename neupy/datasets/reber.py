# -*- coding: utf-8 -*-
import math
from random import choice, randint

import numpy as np


__all__ = ('make_reber', 'is_valid_by_reber', 'make_reber_classification')


avaliable_letters = 'TVPXS'
reber_rules = {
    0: [('T', 1), ('V', 2)],
    1: [('P', 1), ('T', 3)],
    2: [('X', 2), ('V', 4)],
    3: [('X', 2), ('S', None)],
    4: [('P', 3), ('S', None)],
}


def is_valid_by_reber(word):
    """
    Сhecks whether a word belongs to grammar Reber.

    Parameters
    ----------
    word : str or list of letters
        The word that you want to test.

    Returns
    -------
    bool
        ``True`` if word valid by Reber grammar and
        ``False`` otherwise.

    Examples
    --------
    >>> from neupy.datasets import is_valid_by_reber
    >>>
    >>> is_valid_by_reber('TTS')
    True
    >>> is_valid_by_reber('STS')
    False
    >>>
    >>> is_valid_by_reber(['T', 'T', 'S'])
    True
    >>> is_valid_by_reber(['S', 'T', 'S'])
    False
    """
    if not word.endswith("S"):
        return False

    position = 0
    for letter in word:
        possible_letters = reber_rules[position]
        letters = [step[0] for step in possible_letters]
        if letter not in letters:
            return False
        _, position = possible_letters[letters.index(letter)]
    return True


def make_reber(n_words=100):
    """
    Generate list of words valid by Reber grammar.

    Parameters
    ----------
    n_words : int
        Number of reber words, defaults to ``100``.

    Returns
    -------
    list
        List of Reber words.

    Examples
    --------
    >>> from neupy.datasets import make_reber
    >>> make_reber(4)
    ['TPTXVS', 'VXXVS', 'TPPTS', 'TTXVPXXVS']
    """
    if n_words < 1:
        raise ValueError("Must be at least one word")

    words = []
    for i in range(n_words):
        position = 0
        word = []

        while position is not None:
            possible_letters = reber_rules[position]
            letter, position = choice(possible_letters)
            word.append(letter)

        words.append(''.join(word))
    return words


def convert_letters_to_indices(samples):
    """
    Convert Reber Grammar words to the list of indices where
    each index referes to specific letter.

    Parameters
    ----------
    samples : list of str
        List of words.

    Examples
    --------
    >>> convert_letters_to_indices(['XXXXVTTSSV', 'VXXVS'])
    array([array([3, 3, 3, 3, 1, 0, 0, 4, 4, 1]),
           array([1, 3, 3, 1, 4])], dtype=object)
    """
    index_samples = []
    for sample in samples:
        word = [avaliable_letters.index(letter) for letter in sample]
        index_samples.append(np.array(word))
    return np.array(index_samples)


def make_reber_classification(n_samples, invalid_size=0.5, lenrange=(3, 14),
                              return_indices=False):
    """
    Generate random dataset for Reber grammar classification.
    Invalid words contains the same letters as at Reber grammar, but
    they are build without grammar rules.

    Parameters
    ----------
    n_samples : int
        Number of samples in dataset.

    invalid_size : float
        Proportion of invalid words in dataset, defaults to ``0.5``.
        Value must be between ``0`` and ``1``.

    lenrange : tuple
        Length of each word will be bounded by the two numbers
        specified in this range. Defaults to ``(3, 14)``.

    return_indices : bool
        If ``True``, each word will be converted to array where each
        letter converted to the index. Defaults to ``False``.

    Returns
    -------
    tuple
        Return two lists. First contains words and second - labels for them.

    Examples
    --------
    >>> from neupy.datasets import make_reber_classification
    >>>
    >>> data, labels = make_reber_classification(10, invalid_size=0.5)
    >>> data
    array(['SXSXVSXXVX', 'VVPS', 'VVPSXTTS', 'VVS', 'VXVS', 'VVS',
           'PPTTTXPSPTV', 'VTTSXVPTXVXT', 'VSSXSTX', 'TTXVS'],
          dtype='<U12')
    >>> labels
    array([0, 1, 0, 1, 1, 1, 0, 0, 0, 1])
    >>>
    >>> data, labels = make_reber_classification(
    ...     4, invalid_size=0.5, return_indices=True)
    >>> data
    array([array([1, 3, 1, 4]),
           array([0, 3, 0, 3, 0, 4, 3, 0, 4, 4]),
           array([1, 3, 1, 2, 3, 1, 2, 4]),
           array([0, 3, 0, 0, 3, 0, 4, 2, 4, 1, 0, 4, 0])], dtype=object)
    """
    if n_samples < 2:
        raise ValueError("There are must be at least 2 samples")

    if not 0 < invalid_size < 1:
        raise ValueError("`invalid_size` argument value must be between "
                         "zero and one, got {}".format(invalid_size))

    n_valid_words = int(math.ceil(n_samples * invalid_size))
    n_invalid_words = n_samples - n_valid_words

    valid_words = make_reber(n_valid_words)
    valid_labels = [1] * n_valid_words

    invalid_words = []
    invalid_labels = [0] * n_valid_words

    for i in range(n_invalid_words):
        word_length = randint(*lenrange)
        word = [choice(avaliable_letters) for _ in range(word_length)]
        invalid_words.append(''.join(word))

    samples = np.array(valid_words + invalid_words)
    labels = np.array(valid_labels + invalid_labels)

    indices = np.arange(len(samples))
    np.random.shuffle(indices)

    samples, labels = samples[indices], labels[indices]

    if return_indices:
        samples = convert_letters_to_indices(samples)

    return samples, labels
