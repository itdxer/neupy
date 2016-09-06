import numpy as np


__all__ = ('make_digits', 'load_digits')


digits_data = np.array([
    [
        # zero
        0, 1, 1, 0,
        1, 0, 0, 1,
        1, 0, 0, 1,
        1, 0, 0, 1,
        1, 0, 0, 1,
        0, 1, 1, 0,
    ], [
        # one
        0, 1, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
    ], [
        # two
        1, 1, 1, 0,
        0, 0, 0, 1,
        0, 0, 0, 1,
        0, 1, 1, 0,
        1, 0, 0, 0,
        1, 1, 1, 1,
    ], [
        # three
        1, 1, 1, 0,
        0, 0, 0, 1,
        0, 1, 1, 0,
        0, 0, 0, 1,
        0, 0, 0, 1,
        1, 1, 1, 0,
    ], [
        # four
        0, 0, 1, 1,
        0, 1, 0, 1,
        1, 0, 0, 1,
        1, 1, 1, 1,
        0, 0, 0, 1,
        0, 0, 0, 1,
    ], [
        # five
        0, 1, 1, 1,
        1, 0, 0, 0,
        0, 1, 1, 0,
        0, 0, 0, 1,
        0, 0, 0, 1,
        1, 1, 1, 0,
    ], [
        # six
        0, 1, 1, 0,
        1, 0, 0, 0,
        1, 1, 1, 0,
        1, 0, 0, 1,
        1, 0, 0, 1,
        0, 1, 1, 0,
    ], [
        # seven
        0, 1, 1, 1,
        0, 0, 0, 1,
        0, 0, 0, 1,
        0, 0, 1, 0,
        0, 1, 0, 0,
        0, 1, 0, 0,
    ], [
        # eight
        0, 1, 1, 0,
        1, 0, 0, 1,
        0, 1, 1, 0,
        1, 0, 0, 1,
        1, 0, 0, 1,
        0, 1, 1, 0,
    ], [
        # nine
        0, 1, 1, 0,
        1, 0, 0, 1,
        1, 0, 0, 1,
        0, 1, 1, 1,
        0, 0, 0, 1,
        0, 1, 1, 0,
    ]
], dtype=np.uint8)
digits_labels = np.arange(10)


def load_digits():
    """
    Returns dataset that contains discrete digits.

    Returns
    -------
    tuple
        Tuple contains two values. First one is a matrix
        with shape (10, 24). Second one is a vector that contains
        labels for each row. Each digit can be trasnformed
        into (6, 4) binary image.
    """
    return digits_data, digits_labels


def make_digits(noise_level=0.1, n_samples=100):
    """
    Returns discrete digits dataset.

    Parameters
    ----------
    noise_level : float
        Defines level of a discrete noise added to the images.
        Noise level defines probability for the pixel
        to be removed. Value should be in [0, 1) range.
        Defaults to ``0.1``.
    n_samples : int
        Number of samples. Defaults to ``100``.

    Returns
    -------
    tuple
        Tuple contains two values. First one is a matrix
        with shape (n_samples, 24). Second one is a vector
        that contains labels for each row. Each digit can
        be trasnformed into (6, 4) binary image.

    Examples
    --------
    >>> from neupy import datasets, environment
    >>>
    >>> environment.reproducible()
    >>>
    >>> digits, labels = datasets.make_digits(noise_level=0.15)
    >>> digit, label = digits[0], labels[0]
    >>>
    >>> label
    5
    >>>
    >>> digit.reshape((6, 4))
    array([[0, 0, 1, 1],
           [1, 0, 0, 0],
           [0, 1, 1, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1],
           [1, 1, 1, 0]], dtype=uint8)
    """

    if not 0 <= noise_level < 1:
        raise ValueError("noise_level should be float number "
                         "from [0, 1) range, got {}".format(noise_level))

    if n_samples < 1:
        raise ValueError("Number of samples should be an integer greater "
                         "or equal to 1, got {}".format(n_samples))

    digit_indeces = np.random.randint(10, size=n_samples)
    digit_images = digits_data[digit_indeces]
    digit_labels = digits_labels[digit_indeces]

    disable_pixel = np.random.binomial(n=1, p=noise_level,
                                       size=digit_images.shape)
    digit_images[disable_pixel.astype(bool)] = 0

    return digit_images, digit_labels
