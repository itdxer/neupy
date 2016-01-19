""" Code from `tqdm` library: https://github.com/noamraph/tqdm
"""
import sys
import time


__all__ = ('progressbar',)


def format_interval(t):
    mins, seconds = divmod(int(t), 60)
    hours, minutes = divmod(mins, 60)
    if hours:
        return '%d:%02d:%02d' % (hours, minutes, seconds)
    else:
        return '%02d:%02d' % (minutes, seconds)


def format_meter(n, total, elapsed):
    """ Format meter.
    Parameters
    ----------
    n : int
        Number of finished iterations.
    total : int or None
        Total number of iterations.
    elapsed : int
        Number of seconds passed since start
    """
    if n > total:
        total = None

    elapsed_str = format_interval(elapsed)
    rate = '%5.2f' % (n / elapsed) if elapsed else '?'

    if total:
        frac = float(n) / total

        n_bars = 10
        bar_length = int(frac * n_bars)
        bar = '#' * bar_length + '-' * (n_bars - bar_length)

        percentage = '%3d%%' % (frac * 100)

        left_str = format_interval(elapsed / n * (total-n)) if n else '?'

        return '|%s| %d/%d %s [elapsed: %s left: %s, %s iters/sec]' % (
            bar, n, total, percentage, elapsed_str, left_str, rate)

    else:
        return '%d [elapsed: %s, %s iters/sec]' % (n, elapsed_str, rate)


class StatusPrinter(object):
    def __init__(self, file):
        self.file = file
        self.last_printed_len = 0

    def print_status(self, s):
        self.file.write('\r' + s + ' ' * max(self.last_printed_len-len(s), 0))
        self.file.flush()
        self.last_printed_len = len(s)


def progressbar(iterable, desc='', total=None, leave=False, file=sys.stderr,
                mininterval=0.5, miniters=1):
    """ Get an iterable object, and return an iterator which acts
    exactly like the iterable, but prints a progress meter and updates
    it every time a value is requested.

    Parameters
    ----------
    desc : str
        Can contain a short string, describing the progress,
        that is added in the beginning of the line.
    total : int
        Can give the number of expected iterations. If not given,
        len(iterable) is used if it is defined.
    file : object
        Can be a file-like object to output the progress message to.
    leave : bool
        If leave is False, `progressbar` deletes its traces
        from screen after it has finished iterating over all elements.
    mininterval : float
    miniters : int
        If less than mininterval seconds or miniters
        iterations have passed since the last progress meter
        update, it is not updated again.
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    prefix = desc + ': ' if desc else ''

    sp = StatusPrinter(file)
    sp.print_status(prefix + format_meter(0, total, 0))

    start_t = last_print_t = time.time()
    last_print_n = 0
    n = 0
    for obj in iterable:
        yield obj
        # Now the object was created and processed, so we can print the meter.
        n += 1
        if n - last_print_n >= miniters:
            # We check the counter first, to reduce the overhead of time.time()
            cur_t = time.time()
            if cur_t - last_print_t >= mininterval:
                sp.print_status(prefix + format_meter(n, total, cur_t-start_t))
                last_print_n = n
                last_print_t = cur_t

    if not leave:
        sp.print_status('')
        sys.stdout.write('\r')
    else:
        if last_print_n < n:
            cur_t = time.time()
            sp.print_status(prefix + format_meter(n, total, cur_t-start_t))
        file.write('\n')
