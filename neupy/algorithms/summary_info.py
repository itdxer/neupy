from neupy.helpers.table import format_time


__all__ = ('SummaryTable', 'InlineSummary')


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
    def __init__(self, network, table_builder):
        self.network = network
        self.table_builder = table_builder
        table_builder.start()

    def show_last(self):
        training_error = self.network.errors.last()
        validation_error = self.network.validation_errors.last()

        self.table_builder.row([
            self.network.last_epoch,
            training_error if training_error is not None else '-',
            validation_error if validation_error is not None else '-',
            self.network.training.epoch_time,
        ])

    def finish(self):
        self.table_builder.finish()


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
