from __future__ import division

from math import ceil
from time import time
from itertools import groupby
from collections import deque

import six
from numpy import arange, mean
import matplotlib.pyplot as plt

from neupy.utils import format_data, is_row1d
from neupy.helpers import preformat_value
from neupy.core.base import BaseSkeleton
from neupy.core.properties import (Property, FuncProperty, NumberProperty,
                                   BoolProperty)
from neupy.layers import BaseLayer, OutputLayer
from neupy.functions import normilize_error_output, mse
from .utils import iter_until_converge, shuffle
from .connections import (FAKE_CONNECTION, LayerConnection,
                          NetworkConnectionError)


__all__ = ('BaseNetwork',)


def show_training_summary(network):
    network.logs.data("""
        Epoch {epoch}
        Train error:  {error}
        Validation error: {error_out}
        Epoch time: {epoch_time} sec
    """.format(
        epoch=network.epoch,
        error=network.last_error_in() or '-',
        error_out=network.last_error_out() or '-',
        epoch_time=round(network.train_epoch_time, 5)
    ))


def show_epoch_summary(network, show_epoch):
    delay_limit = 1  # in seconds
    prev_summary_time = None
    delay_history_length = 10
    terminal_output_delays = deque(maxlen=delay_history_length)

    while True:
        now = time()

        if prev_summary_time is not None:
            time_delta = now - prev_summary_time
            terminal_output_delays.append(time_delta)

        show_training_summary(network)
        prev_summary_time = now

        if len(terminal_output_delays) == delay_history_length:
            prev_summary_time = None
            average_delay = mean(terminal_output_delays)

            if average_delay < delay_limit:
                show_epoch *= ceil(delay_limit / average_delay)
                network.logs.warning(
                    "Too many outputs in a terminal. Set "
                    "up logging after each {} epoch"
                    "".format(show_epoch)
                )
                terminal_output_delays.clear()

        yield show_epoch


def shuffle_train_data(input_train, target_train):
    if target_train is None:
        return shuffle(input_train), None
    return shuffle(input_train, target_train)


def clean_layers(connection):
    """ Clean layers connections and format transform them into one format.
    Also this function validate layers connections.

    Parameters
    ----------
    connection : list, tuple or object
        Layers connetion in different formats.

    Returns
    -------
    object
        Cleaned layers connection.
    """
    if connection == FAKE_CONNECTION:
        return connection

    if isinstance(connection, tuple):
        connection = list(connection)

    islist = isinstance(connection, list)

    if islist and isinstance(connection[0], BaseLayer):
        chain_connection = connection.pop()
        for layer in reversed(connection):
            chain_connection = LayerConnection(layer, chain_connection)
        connection = chain_connection

    elif islist and isinstance(connection[0], LayerConnection):
        pass

    if not isinstance(connection.output_layer, OutputLayer):
        raise NetworkConnectionError("Final layer must be OutputLayer class "
                                     "instance.")

    return connection


def parse_show_epoch_property(value, n_epochs):
    if isinstance(value, int):
        return value

    number_end_position = value.index('time')
    # Ignore grammar mistakes like `2 time`, this error could be
    # really annoying
    n_epochs_to_check = int(value[:number_end_position].strip())

    if n_epochs <= n_epochs_to_check:
        return 1

    return int(round(n_epochs / n_epochs_to_check))


class ShowEpochProperty(Property):
    """ Class helps validate specific syntax for `show_epoch`
    property from ``BaseNetwork`` class.
    """
    expected_type = tuple([int] + [six.string_types])

    def validate(self, value):
        if not isinstance(value, six.string_types):
            if value < 1:
                raise ValueError("Property `{}` value should be integer "
                                 "greater than zero or string. See the "
                                 "documentation for more information."
                                 "".format(self.name))
            return

        if 'time' not in value:
            raise ValueError("`{}` value has invalid string format."
                             "".format(self.name))

        valid_endings = ('times', 'time')
        number_end_position = value.index('time')
        number_part = value[:number_end_position].strip()

        if not value.endswith(valid_endings) or not number_part.isdigit():
            valid_endings_formated = ', '.join(valid_endings)
            raise ValueError(
                "Property `{}` in string format should be a positive number "
                "with one of those endings: {}. For example: `10 times`."
                "".format(self.name, valid_endings_formated)
            )

        if int(number_part) < 1:
            raise ValueError("Part that related to the number in `{}` "
                             "property should be an integer greater or "
                             "equal to one.".format(self.name))


class BaseNetwork(BaseSkeleton):
    """ Base class Network algorithms.

    Parameters
    ----------
    {full_params}

    Methods
    -------
    {plot_errors}
    {last_error}
    """
    error = FuncProperty(default=mse)
    use_bias = BoolProperty(default=True)
    step = NumberProperty(default=0.1)

    # Training settings
    show_epoch = ShowEpochProperty(min_size=1, default='10 times')
    shuffle_data = BoolProperty(default=False)

    # Signals
    train_epoch_end_signal = FuncProperty()
    train_end_signal = FuncProperty()

    def __init__(self, connection, **options):
        self.connection = clean_layers(connection)

        self.errors_in = []
        self.errors_out = []
        self.train_epoch_time = None

        self.layers = list(self.connection)
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]
        self.train_layers = self.layers[:-1]

        super(BaseNetwork, self).__init__(**options)
        self.setup_defaults()

        if self.verbose:
            self.show_network_options(highlight_options=options)

        self.init_layers()

    def show_network_options(self, highlight_options=None):
        available_classes = [cls.__name__ for cls in self.__class__.__mro__]
        logs = self.logs

        if highlight_options is None:
            highlight_options = {}

        def classname_grouper(option):
            classname = option[1].class_name
            class_priority = -available_classes.index(classname)
            return (class_priority, classname)

        # Sort and group options by classes
        grouped_options = groupby(
            sorted(self.options.items(), key=classname_grouper),
            key=classname_grouper
        )

        if isinstance(self.connection, LayerConnection):
            logs.header("Network structure")
            logs.log("LAYERS", self.connection)

        # Just display in terminal all network options.
        logs.header("Network options")
        for (_, clsname), class_options in grouped_options:
            if not class_options:
                # When in some class we remove all available attributes
                # we just skip it.
                continue

            logs.simple("{}:".format(clsname))

            for key, data in sorted(class_options):
                if key in highlight_options:
                    logger = logs.log
                    value = highlight_options[key]
                else:
                    logger = logs.gray_log
                    value = data.value

                logger("OPTION", "{} = {}".format(
                    key, preformat_value(value))
                )
            logs.empty()

    def setup_defaults(self):
        """ Setup default values before populate the options.
        """

    # ----------------- Neural Network Layers ---------------- #

    def init_layers(self):
        """ Initialize layers.
        """
        if self.connection == FAKE_CONNECTION:
            return

        for layer in self.train_layers:
            layer.initialize(with_bias=self.use_bias)

    # ----------------- Neural Network Train ---------------- #

    def _train(self, input_train, target_train=None, input_test=None,
               target_test=None, epochs=100, epsilon=None):
        """ Main method for the Neural Network training.
        """

        # ----------- Pre-format target data ----------- #

        input_row1d = is_row1d(self.input_layer)
        input_train = format_data(input_train, row1d=input_row1d)

        target_row1d = is_row1d(self.output_layer)
        target_train = format_data(target_train, row1d=target_row1d)

        if input_test is not None:
            input_test = format_data(input_test, row1d=input_row1d)

        if target_test is not None:
            target_test = format_data(target_test, row1d=target_row1d)

        # ----------- Validate input values ----------- #

        if epsilon is not None and epochs <= 2:
            raise ValueError("Network should train at teast 3 epochs before "
                             "check the difference between errors")

        # ----------- Predefine parameters ----------- #

        self.epoch = 1
        show_epoch = self.show_epoch
        logs = self.logs
        compute_error_out = (input_test is not None and
                             target_test is not None)
        predict = self.predict
        last_epoch_shown = 0

        if epsilon is not None:
            iterepochs = iter_until_converge(self, epsilon, max_epochs=epochs)

            if isinstance(show_epoch, six.string_types):
                show_epoch = 100
                logs.warning("Can't use `show_epoch` value in converging "
                             "mode. Set up 100 to `show_epoch` property "
                             "by default.")

        else:
            iterepochs = range(1, epochs + 1)
            show_epoch = parse_show_epoch_property(show_epoch, epochs)

        epoch_summary = show_epoch_summary(self, show_epoch)

        # ----------- Train process ----------- #

        logs.header("Start train")
        logs.log("TRAIN", "Train data size: {}".format(input_train.shape[0]))

        if input_test is not None:
            logs.log("TRAIN", "Validation data size: {}"
                              "".format(input_test.shape[0]))

        if epsilon is None:
            logs.log("TRAIN", "Total epochs: {}".format(epochs))
        else:
            logs.log("TRAIN", "Max epochs: {}".format(epochs))

        logs.empty()

        # Optimizations for long loops
        errors = self.errors_in
        errors_out = self.errors_out
        shuffle_data = self.shuffle_data

        error_func = self.error
        train_epoch = self.train_epoch
        train_epoch_end_signal = self.train_epoch_end_signal
        train_end_signal = self.train_end_signal

        self.input_train = input_train
        self.target_train = target_train

        for epoch in iterepochs:
            self.epoch = epoch
            epoch_start_time = time()

            if shuffle_data:
                input_train, target_train = shuffle_train_data(input_train,
                                                               target_train)
                self.input_train = input_train
                self.target_train = target_train

            try:
                error = train_epoch(input_train, target_train)

                if compute_error_out:
                    predicted_test = predict(input_test)
                    error_out = error_func(predicted_test, target_test)
                    errors_out.append(error_out)

                errors.append(error)
                self.train_epoch_time = time() - epoch_start_time

                if epoch % show_epoch == 0 or epoch == 1:
                    show_epoch = next(epoch_summary)
                    last_epoch_shown = epoch

                if train_epoch_end_signal is not None:
                    train_epoch_end_signal(self)

            except StopIteration as err:
                logs.log("TRAIN", "Epoch #{} stopped. {}"
                                  "".format(epoch, str(err)))
                break

        if epoch != last_epoch_shown:
            show_training_summary(self)

        if train_end_signal is not None:
            train_end_signal(self)

        logs.log("TRAIN", "End train")

    # ----------------- Errors ----------------- #

    def _last_error(self, errors):
        if errors and errors[-1] is not None:
            return normilize_error_output(errors[-1])

    def last_error_in(self):
        return self._last_error(self.errors_in)

    def last_error(self):
        return self._last_error(self.errors_in)

    def last_error_out(self):
        return self._last_error(self.errors_out)

    def previous_error(self):
        errors = self.errors_in
        return normilize_error_output(errors[-2]) if len(errors) > 2 else None

    def _normalized_errors(self, errors):
        if not len(errors) or isinstance(errors[0], float):
            return errors

        self.logs.warning("Your errors bad formated for plot output. "
                          "They will be normilized.")

        normilized_errors = []
        for error in errors:
            normilized_errors.append(normilize_error_output(error))

        return normilized_errors

    def normalized_errors_in(self):
        return self._normalized_errors(self.errors_in)

    def normalized_errors_out(self):
        return self._normalized_errors(self.errors_out)

    def plot_errors(self, logx=False):
        if not self.errors_in:
            return

        errors_in = self.normalized_errors_in()
        errors_out = self.normalized_errors_out()
        errors_range = arange(len(errors_in))
        plot_function = plt.semilogx if logx else plt.plot

        line_error_in, = plot_function(errors_range, errors_in)
        title_text = 'Learning error after each epoch'

        if errors_out:
            line_error_out, = plot_function(errors_range, errors_out)
            plt.legend(
                [line_error_in, line_error_out],
                ['Train error', 'Validation error']
            )
            title_text = 'Learning errors after each epoch'

        plt.title(title_text)
        plt.xlim(0)

        plt.ylabel('Error')
        plt.xlabel('Epoch')

        plt.show()

    # ----------------- Representations ----------------- #

    def get_class_name(self):
        return self.__class__.__name__

    def __repr__(self):
        classname = self.get_class_name()
        options_repr = self._repr_options()

        if self.connection != FAKE_CONNECTION:
            return "{}({}, {})".format(classname, self.connection,
                                       options_repr)
        return "{}({})".format(classname, options_repr)
