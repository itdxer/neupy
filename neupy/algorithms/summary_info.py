# -*- coding: utf-8 -*-
from __future__ import unicode_literals


__all__ = ('SummaryTable', 'InlineSummary')


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
        # microseconds
        return '{:.0f} μs'.format(time * 1e6)

    # nanoseconds or smaller
    return '{:.0f} ns'.format(time * 1e9)


class SummaryTable(object):
    """
    Class that shows network's training errors in the
    form of a table.

    Parameters
    ----------
    network : BaseNetwork
        Network instance.

    table_builder : TableBuilder
        Pre-defined table builder with specified table structure.

    delay_limit : float
        Defaults to ``1``.

    delay_history_length : int
        Defaults to ``10``.
    """
    def __init__(self, network, columns):
        self.network = network
        self.columns = columns
        self.logs = self.network.logs
        self.logs.table_header(columns)
        self.finished = False

    def show_last(self):
        training_error = self.network.errors.last()
        validation_error = self.network.validation_errors.last()

        self.logs.table_row([
            self.network.last_epoch,
            training_error if training_error is not None else '-',
            validation_error if validation_error is not None else '-',
            format_time(self.network.training.epoch_time),
        ])

    def finish(self):
        if not self.finished:
            self.logs.table_bottom(len(self.columns))
            self.finished = True

        self.logs.newline()


class InlineSummary(object):
    """
    Class that shows network's training errors in the
    form of a table.

    Parameters
    ----------
    network : BaseNetwork
        Network instance.
    """
    def __init__(self, network):
        self.network = network

    def show_last(self):
        network = self.network
        logs = network.logs

        train_error = network.errors.last()
        validation_error = network.validation_errors.last()
        epoch_training_time = format_time(network.training.epoch_time)

        if validation_error is not None:
            logs.write(
                "epoch #{}, train err: {:.6f}, valid err: {:.6f}, time: {}"
                "".format(network.last_epoch, train_error, validation_error,
                          epoch_training_time))
        else:
            logs.write(
                "epoch #{}, train err: {:.6f}, time: {}"
                "".format(network.last_epoch, train_error,
                          epoch_training_time))

    def finish(self):
        pass
