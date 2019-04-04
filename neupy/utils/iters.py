from __future__ import division

import math

import numpy as np
import progressbar

from neupy.utils.misc import as_tuple


__all__ = (
    'apply_batches', 'minibatches',
    'count_minibatches', 'count_samples',
)


def count_samples(inputs):
    if isinstance(inputs, (list, tuple)):
        return count_samples(inputs[0])
    return len(inputs)


def count_minibatches(inputs, batch_size):
    return int(math.ceil(count_samples(inputs) / batch_size))


def apply_slices(inputs, indices):
    if inputs is None:
        return inputs

    if isinstance(inputs, (list, tuple)):
        return [apply_slices(input_, indices) for input_ in inputs]

    return inputs[indices]


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
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for index in range(n_batches):
            batch_slice = slice(index * batch_size, (index + 1) * batch_size)
            yield apply_slices(inputs, indices[batch_slice])

    elif n_batches != 1:
        for index in range(n_batches):
            batch_slice = slice(index * batch_size, (index + 1) * batch_size)
            yield apply_slices(inputs, batch_slice)

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
                  show_output=False, average_outputs=False):
    """
    Splits inputs into mini-batches and passes them to the function.
    Function returns list of outputs or average loss in case
    if ``average_outputs=True``.

    Parameters
    ----------
    function : func
        Function that accepts one or more positional inputs.
        Each of them should be an array-like variable that
        have exactly the same number of rows.

    inputs : tuple, list
        The arguments that will be provided to the function specified
        in the ``function`` argument.

    batch_size : int
        Mini-batch size. Defines maximum number of samples that will be
        used as an input to the ``function``.

    show_progressbar : bool
        When ``True`` than progress bar will be shown in the terminal.
        Defaults to ``False``.

    show_output : bool
        Assumes that outputs from the function errors. The ``True`` value
        will show information in the progressbar. Error will be related to
        the last epoch. Defaults to ``False``.

    average_outputs : bool
        Output from each batch will be combined into single average. This
        option assumes that loss per batch was calculated from.
        Defaults to ``False``.

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

    # Clean progressbar from the screen
    bar.fd.write('\r' + ' ' * bar.term_width + '\r')

    if average_outputs:
        # When loss calculated per batch separately it might be
        # necessary to combine error into single value
        return average_batch_errors(outputs, n_samples, batch_size)

    return outputs
