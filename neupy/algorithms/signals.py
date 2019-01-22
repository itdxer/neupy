# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import time

import progressbar

from neupy.utils import iters


__all__ = ('ProgressbarSignal', 'PrintLastErrorSignal', 'EpochEndSignal',
           'ErrorCollector')


def format_time(time):
    """
    Format seconds into human readable format.

    Parameters
    ----------
    time : float
        Time specified in seconds

    Returns
    -------
    str
        Formated time.
    """
    mins, seconds = divmod(int(time), 60)
    hours, minutes = divmod(mins, 60)

    if hours > 0:
        return '{:0>2d}:{:0>2d}:{:0>2d}'.format(hours, minutes, seconds)

    elif minutes > 0:
        return '{:0>2d}:{:0>2d}'.format(minutes, seconds)

    elif seconds > 0:
        return '{:.0f} sec'.format(seconds)

    elif time >= 1e-3:
        return '{:.0f} ms'.format(time * 1e3)

    elif time >= 1e-6:
        return '{:.0f} μs'.format(time * 1e6)  # microseconds

    return '{:.0f} ns'.format(time * 1e9)  # nanoseconds or smaller


class EpochEndSignal(object):
    def __init__(self, function):
        self.function = function

    def epoch_end(self, network):
        self.function(network)


class ProgressbarSignal(object):
    def train_start(self, network, **kwargs):
        if kwargs['batch_size'] is None:
            self.n_batches = 1
            return

        self.n_batches = iters.count_minibatches(
            kwargs['X_train'],
            kwargs['batch_size'])

    def epoch_start(self, network):
        self.index = 0
        self.last_train_error = None
        self.bar = progressbar.NullBar()

        if network.verbose and self.n_batches >= 2:
            self.bar = iters.make_progressbar(self.n_batches, show_output=True)
            self.bar.update(0)

    def train_error(self, network, **data):
        self.last_train_error = data['value']

    def update_end(self, network):
        self.index += 1
        self.bar.update(self.index, loss=self.last_train_error)

    def epoch_end(self, network):
        self.bar.finish(end='\r')
        self.bar.fd.write('\r' + ' ' * self.bar.term_width + '\r')

    def __reduce__(self):
        return self.__class__, tuple()


class PrintLastErrorSignal(object):
    def print_last_error(self, network):
        messages = []
        base_message = "#{} : [{}] ".format(
            network.last_epoch,
            format_time(self.last_epoch_time))

        if self.last_train_error is not None:
            messages.append("train: {:.6f}".format(self.last_train_error))

        if self.last_valid_error is not None:
            messages.append("valid: {:.6f}".format(self.last_valid_error))

        network.logs.write(base_message + ', '.join(messages))

    def train_start(self, network, **kwargs):
        self.first_epoch = network.last_epoch + 1
        self.last_epoch_shown = 0

    def epoch_start(self, network):
        self.epoch_start_time = time.time()
        self.last_train_error = None
        self.last_valid_error = None

    def train_error(self, network, value, **kwargs):
        self.last_train_error = value

    def valid_error(self, network, value, **kwargs):
        self.last_valid_error = value

    def epoch_end(self, network):
        epoch = network.last_epoch
        self.last_epoch_time = time.time() - self.epoch_start_time

        if epoch % network.show_epoch == 0 or epoch == self.first_epoch:
            self.print_last_error(network)
            self.last_epoch_shown = network.last_epoch

    def train_end(self, network):
        if network.last_epoch != self.last_epoch_shown:
            self.print_last_error(network)


class ErrorCollector(object):
    def __init__(self):
        self.valid = []
        self.train = []

    def train_error(self, network, value, **kwargs):
        self.train.append(value)

    def valid_error(self, network, value, **kwargs):
        self.valid.append(value)
