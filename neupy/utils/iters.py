from __future__ import division

import math

import numpy as np
import progressbar

from neupy.utils.misc import as_tuple


__all__ = ('apply_batches', 'minibatches')


def count_samples(inputs):
    if isinstance(inputs, (list, tuple)):
        return count_samples(inputs[0])
    return len(inputs)


def count_minibatches(inputs, batch_size):
    return int(math.ceil(count_samples(inputs) / batch_size))


def apply_slices(inputs, indeces):
    if inputs is None:
        return inputs

    if isinstance(inputs, (list, tuple)):
        return [apply_slices(input_, indeces) for input_ in inputs]

    return inputs[indeces]


def minibatches(inputs, batch_size=None, shuffle=False):
    """
    Iterates batch slices.

    Parameters
    ----------
    inputs : array-like, list

    batch_size : int
        Mini-batch size. Number should be greater than ``0``.

    shuffle : bool
        Defaults to ``True``.

    Yields
    ------
    object
        Batch slices.
    """
    n_samples = count_samples(inputs)
    batch_size = n_samples if batch_size is None else batch_size
    n_batches = count_minibatches(inputs, batch_size)

    if shuffle:
        indeces = np.arange(n_samples)
        np.random.shuffle(indeces)

        for index in range(n_batches):
            batch_indeces = slice(index * batch_size, (index + 1) * batch_size)
            yield apply_slices(inputs, indeces[batch_indeces])

    elif n_batches != 1:
        for index in range(n_batches):
            batch_indeces = slice(index * batch_size, (index + 1) * batch_size)
            yield apply_slices(inputs, batch_indeces)

    else:
        yield inputs


def average_batch_errors(errors, n_samples, batch_size):
    """
    Computes average error per sample. Function assumes that error from
    each batch was just an average loss of each individual sample.

    Parameters
    ----------
    errors : list
        List of average errors calculated per batch.

    n_samples : int
        Number of input samples..

    batch_size : int
        Mini-batch size.

    Returns
    -------
    float
        Average error per sample.
    """
    errors = np.atleast_1d(errors)
    batches = np.ones_like(errors) * batch_size

    if len(errors) == 1:
        return errors.item(0)

    if n_samples % batch_size != 0:
        # Last batch can be smaller than the usual one, because we
        # won't have enough samples for the full one.
        batches[-1] = n_samples % batch_size

    return np.dot(errors, batches) / n_samples


def make_progressbar(max_value, show_output):
    widgets = [
        progressbar.Timer(format='Time: %(elapsed)s'),
        ' |',
        progressbar.Percentage(),
        progressbar.Bar(),
        ' ',
        progressbar.ETA(),
    ]

    if show_output:
        widgets.extend([' | ', progressbar.DynamicMessage('loss')])

    return progressbar.ProgressBar(max_value=max_value, widgets=widgets)


def apply_batches(function, inputs, batch_size, show_progressbar=False,
                  show_output=True, average_outputs=False):
    """
    Apply batches to a specified function.

    Parameters
    ----------
    function : func
        Function that accepts one or more positional inputs.
        Each of them should be an array-like variable that
        have exactly the same number of rows.

    inputs : tuple, list
        The arguemnts that will be provided to the function specified
        in the ``function`` argument.

    batch_size : int
        Mini-batch size.

    show_output : bool
        Assumes that outputs from the function errors.
        ``True`` will show information in the progressbar.
        Error will be related to the last epoch.
        Defaults to ``True``.

    Returns
    -------
    list
        List of function outputs.
    """
    n_samples = count_samples(inputs)
    batch_size = n_samples if batch_size is None else batch_size

    n_batches = count_minibatches(inputs, batch_size)
    bar = progressbar.NullBar()

    if show_progressbar and n_batches >= 2:
        bar = make_progressbar(n_batches, show_output)
        bar.update(0)  # triggers empty progressbar

    outputs = []
    iterator = minibatches(inputs, batch_size, shuffle=False)

    for i, sliced_inputs in enumerate(iterator):
        output = function(*as_tuple(sliced_inputs))
        outputs.append(output)

        kwargs = dict(loss=output) if show_output else {}
        bar.update(i, **kwargs)

    bar.fd.write('\r' + ' ' * bar.term_width + '\r')

    if average_outputs:
        return average_batch_errors(outputs, n_samples, batch_size)

    return outputs
