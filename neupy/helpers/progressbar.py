from __future__ import division

import sys
import time
import collections


__all__ = ('Progressbar',)


def format_time(time_in_seconds):
    """
    Format seconds into human readable format.

    Parameters
    ----------
    time_in_seconds : float

    Returns
    -------
    str
        Formated time.
    """
    mins, seconds = divmod(int(time_in_seconds), 60)
    hours, minutes = divmod(mins, 60)

    if hours > 0:
        return '{:0>2d}:{:0>2d}:{:0>2d}'.format(hours, minutes, seconds)

    return '{:0>2d}:{:0>2d}'.format(minutes, seconds)


class FormatInlineDict(collections.OrderedDict):
    """
    Simple class that inherits all functionality from
    ``collections.OrderedDict`` class. It provides one
    additional method that makes inline formatting.
    """
    def __format__(self, format):
        formated_values = []
        for key, value in self.items():
            formated_value = '{}: {}'.format(key, value)
            formated_values.append(formated_value)
        return ' '.join(formated_values)


class Progressbar(collections.Iterable):
    """
    Get an iterable object, and return an iterator which acts
    exactly like the iterable, but prints a progress meter and updates
    it every time a value is requested.

    Parameters
    ----------
    iterable : list, tuple
    update_freq : float
        Says how often to update progressbar (is seconds).
        Useful in case if iterations are very fast.
        Defaults to ``0.1``.
    description : str
        Can contain a short string, describing the progress,
        that is added in the beginning of the line.
    file : object
        Can be a file-like object to output the progress message to.
    """
    def __init__(self, iterable, update_freq=0.1, description='',
                 file=sys.stderr):

        self.iterable = iterable
        self.file = file
        self.description = description
        self.update_freq = update_freq

        self.total = len(iterable)
        self.last_printed_len = 0
        self.show_in_next_iteration = {}

    def write(self, text):
        n_spaces = max(self.last_printed_len - len(text), 0)

        self.file.write('\r' + text + ' ' * n_spaces)
        self.file.flush()

        self.last_printed_len = len(text)

    def clean(self):
        self.write('')
        self.file.write('\r')

    def update_status(self, n_finished, elapsed):
        """
        Update progressbar status in the specified IO file.

        Parameters
        ----------
        n_finished : int
            Number of finished iterations.
        elapsed : int
            Number of seconds passed since start
        """
        n_bars = 10
        n_total = self.total

        ratio_finished = n_finished / n_total
        bar_length = int(ratio_finished * n_bars)

        if n_finished:
            left_time = elapsed * (n_total - n_finished) / n_finished
            left_str = format_time(left_time)
        else:
            left_str = '?'

        information = FormatInlineDict([
            ('elapsed', format_time(elapsed)),
            ('left', left_str),
        ])
        information.update(self.show_in_next_iteration)

        self.write(
            '{desc}{sep}|{progress:-<10s}| {finished}/{total} '
            '{percent:>4.0%} [{information}]'.format(
                desc=self.description,
                sep=(': ' if self.description else ''),
                progress='#' * bar_length,
                finished=n_finished,
                total=n_total,
                percent=ratio_finished,
                information=information
            )
        )
        self.show_in_next_iteration = {}

    def __iter__(self):
        start_time = time.time()
        update_freq = self.update_freq

        self.last_printed_len = 0
        self.update_status(n_finished=0, elapsed=0)

        last_update_time = start_time

        try:
            for i, element in enumerate(self.iterable, start=1):
                yield element

                current_time = time.time()

                if current_time - last_update_time > update_freq:
                    last_update_time = current_time
                    time_delta = current_time - start_time
                    self.update_status(n_finished=i, elapsed=time_delta)

        finally:
            self.clean()
