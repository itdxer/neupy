import math
import time
from abc import ABCMeta, abstractmethod
from collections import deque

import six
import numpy as np

from neupy.helpers.table import format_time


__all__ = ('SummaryTable', 'InlineSummary')


class BaseSummary(six.with_metaclass(ABCMeta)):
    @abstractmethod
    def show_last(self, network):
        pass

    @abstractmethod
    def finish(self):
        pass


class SummaryTable(BaseSummary):
    """ Class that shows network's training in the formatted
    table.

    Parameters
    ----------
    network : BaseNetwork
        Network instance.
    table_builder : TableBuilder
        Pre-defined table builder with specified table
        structure.
    delay_limit : float
    delay_history_length : int
    """
    def __init__(self, network, table_builder, delay_limit=1.,
                 delay_history_length=10):

        self.network = network
        self.table_builder = table_builder
        self.delay_limit = delay_limit
        self.delay_history_length = delay_history_length

        self.prev_summary_time = None
        self.terminal_output_delays = deque(maxlen=delay_history_length)

        table_builder.start()

    def show_last(self):
        network = self.network
        table_builder = self.table_builder
        terminal_output_delays = self.terminal_output_delays

        now = time.time()

        if self.prev_summary_time is not None:
            time_delta = now - self.prev_summary_time
            terminal_output_delays.append(time_delta)

        table_builder.row([
            network.last_epoch,
            network.errors.last() or '-',
            network.validation_errors.last() or '-',
            network.training.epoch_time,
        ])
        self.prev_summary_time = now

        if len(terminal_output_delays) == self.delay_history_length:
            self.prev_summary_time = None
            average_delay = np.mean(terminal_output_delays)

            if average_delay < self.delay_limit:
                show_epoch = int(
                    network.training.show_epoch *
                    math.ceil(self.delay_limit / average_delay)
                )
                table_builder.message("Too many outputs in the terminal. "
                                      "Set up logging after each {} epochs"
                                      "".format(show_epoch))

                terminal_output_delays.clear()
                network.training.show_epoch = show_epoch

    def finish(self):
        self.table_builder.finish()


class InlineSummary(BaseSummary):
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
                          epoch_training_time)
            )
        else:
            logs.write(
                "epoch #{}, train err: {:.6f}, time: {}"
                "".format(network.last_epoch, train_error,
                          epoch_training_time)
            )

    def finish(self):
        pass
