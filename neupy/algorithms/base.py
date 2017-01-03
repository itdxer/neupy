from __future__ import division, absolute_import

import time
import types
from itertools import groupby

import six
import numpy as np

from neupy.utils import preformat_value, AttributeKeyDict
from neupy.helpers import table
from neupy.exceptions import StopTraining
from neupy.core.base import BaseSkeleton
from neupy.core.properties import (BoundedProperty, NumberProperty,
                                   Property)
from .summary_info import SummaryTable, InlineSummary
from .utils import iter_until_converge, shuffle


__all__ = ('BaseNetwork',)


def show_network_options(network, highlight_options=None):
    """
    Display all available parameters options for Neural Network.

    Parameters
    ----------
    network : object
        Neural network instance.

    highlight_options : list
        List of enabled options. In that case all options from that
        list would be marked with a green color.
    """
    available_classes = [cls.__name__ for cls in network.__class__.__mro__]
    logs = network.logs

    if highlight_options is None:
        highlight_options = {}

    def group_by_class_name(value):
        _, option = value
        option_priority = -available_classes.index(option.class_name)
        return option_priority, option.class_name

    grouped_options = groupby(
        sorted(network.options.items(), key=group_by_class_name),
        group_by_class_name,
    )

    logs.title("Main information")
    logs.message("ALGORITHM", network.class_name())
    logs.newline()

    for (_, class_name), options in grouped_options:
        for key, data in sorted(options):
            if key in highlight_options:
                msg_color = 'green'
                value = highlight_options[key]
            else:
                msg_color = 'gray'
                value = data.value

            formated_value = preformat_value(value)
            msg_text = "{} = {}".format(key, formated_value)
            logs.message("OPTION", msg_text, color=msg_color)

    logs.newline()


def logging_info_about_the_data(network, input_train, input_test):
    logs = network.logs
    training_shapes = preformat_value(input_train)

    logs.title("Start training")
    logs.message("TRAINING DATA", "shapes: {}".format(training_shapes))

    if isinstance(training_shapes[0], int):
        training_shapes = [training_shapes]

    if input_test is not None:
        test_shapes = preformat_value(input_test)
        logs.message("TEST DATA", "shapes: {}".format(test_shapes))

        if isinstance(test_shapes[0], int):
            test_shapes = [test_shapes]

        for training_shape, test_shape in zip(training_shapes, test_shapes):
            if training_shape[1:] != test_shape[1:]:
                raise ValueError(
                    "Train and test samples should have the same feature "
                    "shapes. Got training input with shape {} and test "
                    "input with shape {}".format(training_shape, test_shape))


def logging_info_about_training(network, epochs, epsilon):
    logs = network.logs
    if epsilon is None:
        logs.message("TRAINING", "Total epochs: {}".format(epochs))
    else:
        logs.message("TRAINING", "Epsilon: {}, Max epochs: {}"
                                 "".format(epsilon, epochs))


def parse_show_epoch_property(network, n_epochs, epsilon=None):
    show_epoch = network.show_epoch

    if isinstance(show_epoch, int):
        return show_epoch

    if epsilon is not None and isinstance(show_epoch, six.string_types):
        network.logs.warning("Can't use `show_epoch` value in converging "
                             "mode. Set up `show_epoch` property equal to 1")
        return 1

    number_end_position = show_epoch.index('time')
    # Ignore grammar mistakes like `2 time`, this error could be
    # really annoying
    n_epochs_to_check = int(show_epoch[:number_end_position].strip())

    if n_epochs <= n_epochs_to_check:
        return 1

    return int(round(n_epochs / n_epochs_to_check))


def create_training_epochs_iterator(network, epochs, epsilon=None):
    if epsilon is not None:
        return iter_until_converge(network, epsilon, max_epochs=epochs)

    next_epoch = network.last_epoch + 1
    return range(next_epoch, next_epoch + epochs)


class ShowEpochProperty(BoundedProperty):
    """
    Class helps validate specific syntax for `show_epoch`
    property from ``BaseNetwork`` class.

    Parameters
    ----------
    {BoundedProperty.Parameters}
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


class ErrorHistoryList(list):
    """
    Wrapper around the built-in list class that adds a few
    additional methods.
    """
    def last(self):
        """
        Returns last element if list is not empty,
        ``None`` otherwise.
        """
        if self and self[-1] is not None:
            return np.sum(self[-1])

    def previous(self):
        """
        Returns last element if list is not empty,
        ``None`` otherwise.
        """
        if len(self) >= 2 and self[-2] is not None:
            return np.sum(self[-2])

    def normalized(self):
        """
        Normalize list that contains error outputs.

        Returns
        -------
        list
            Return the same list with normalized values if there
            where some problems.
        """
        if not self:
            return self

        normalized_errors = map(np.sum, self)
        return ErrorHistoryList(normalized_errors)


class BaseNetwork(BaseSkeleton):
    """
    Base class for Neural Network algorithms.

    Parameters
    ----------
    step : float
        Learning rate, defaults to ``0.1``.

    show_epoch : int or str
        This property controls how often the network will
        display information about training. There are two
        main syntaxes for this property.

        - You can define it as a positive integer number. It
          defines how offen would you like to see summary
          output in terminal. For instance, number `100` mean
          that network shows summary at 100th, 200th,
          300th ... epochs.

        - String defines number of times you want to see output in
          terminal. For instance, value ``'2 times'`` mean that
          the network will show output twice with approximately
          equal period of epochs and one additional output would
          be after the finall epoch.

        Defaults to ``1``.

    shuffle_data : bool
        If it's ``True`` class shuffles all your training data before
        training your network, defaults to ``True``.

    epoch_end_signal : function
        Calls this function when train epoch finishes.

    train_end_signal : function
        Calls this function when train process finishes.

    {Verbose.Parameters}

    Attributes
    ----------
    errors : ErrorHistoryList
        Contains list of training errors. This object has the same
        properties as list and in addition there are three additional
        useful methods: `last`, `previous` and `normalized`.

    train_errors : ErrorHistoryList
        Alias to the ``errors`` attribute.

    validation_errors : ErrorHistoryList
        The same as `errors` attribute, but it contains only validation
        errors.

    last_epoch : int
        Value equals to the last trained epoch. After initialization
        it is equal to ``0``.
    """
    step = NumberProperty(default=0.1, minval=0)

    show_epoch = ShowEpochProperty(minval=1, default=1)
    shuffle_data = Property(default=False, expected_type=bool)

    epoch_end_signal = Property(expected_type=types.FunctionType)
    train_end_signal = Property(expected_type=types.FunctionType)

    def __init__(self, *args, **options):
        self.errors = self.train_errors = ErrorHistoryList()
        self.validation_errors = ErrorHistoryList()
        self.training = AttributeKeyDict()
        self.last_epoch = 0

        super(BaseNetwork, self).__init__(*args, **options)

        if self.verbose:
            show_network_options(self, highlight_options=options)

    def predict(self, input_data):
        """
        Return prediction results for the input data.

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        array-like
        """
        raise NotImplementedError

    def on_epoch_start_update(self, epoch):
        """
        Function would be trigger before run all training procedure
        related to the current epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        self.last_epoch = epoch

    def train_epoch(self, input_train, target_train=None):
        raise NotImplementedError()

    def prediction_error(self, input_test, target_test):
        raise NotImplementedError()

    def train(self, input_train, target_train=None, input_test=None,
              target_test=None, epochs=100, epsilon=None,
              summary='table'):
        """
        Method train neural network.

        Parameters
        ----------
        input_train : array-like

        target_train : array-like or None

        input_test : array-like or None

        target_test : array-like or None

        epochs : int
            Defaults to `100`.

        epsilon : float or None
            Defaults to ``None``.
        """
        show_epoch = self.show_epoch
        logs = self.logs
        training = self.training = AttributeKeyDict()

        if epochs <= 0:
            raise ValueError("Number of epochs needs to be greater than 0.")

        if epsilon is not None and epochs <= 2:
            raise ValueError("Network should train at teast 3 epochs before "
                             "check the difference between errors")

        logging_info_about_the_data(self, input_train, input_test)
        logging_info_about_training(self, epochs, epsilon)
        logs.newline()

        if summary == 'table':
            summary = SummaryTable(
                table_builder=table.TableBuilder(
                    table.Column(name="Epoch #"),
                    table.NumberColumn(name="Train err", places=4),
                    table.NumberColumn(name="Valid err", places=4),
                    table.TimeColumn(name="Time", width=10),
                    stdout=logs.write
                ),
                network=self,
                delay_limit=1.,
                delay_history_length=10,
            )

        elif summary == 'inline':
            summary = InlineSummary(network=self)

        else:
            raise ValueError("`{}` is unknown summary type"
                             "".format(summary))

        iterepochs = create_training_epochs_iterator(self, epochs, epsilon)
        show_epoch = parse_show_epoch_property(self, epochs, epsilon)
        training.show_epoch = show_epoch

        # Storring attributes and methods in local variables we prevent
        # useless __getattr__ call a lot of times in each loop.
        # This variables speed up loop in case on huge amount of
        # iterations.
        training_errors = self.errors
        validation_errors = self.validation_errors
        shuffle_data = self.shuffle_data

        train_epoch = self.train_epoch
        epoch_end_signal = self.epoch_end_signal
        train_end_signal = self.train_end_signal
        on_epoch_start_update = self.on_epoch_start_update

        is_first_iteration = True
        can_compute_validation_error = (input_test is not None)
        last_epoch_shown = 0

        with logs.disable_user_input():
            for epoch in iterepochs:
                validation_error = None
                epoch_start_time = time.time()
                on_epoch_start_update(epoch)

                if shuffle_data:
                    input_train, target_train = shuffle(input_train,
                                                        target_train)
                try:
                    train_error = train_epoch(input_train, target_train)

                    if can_compute_validation_error:
                        validation_error = self.prediction_error(input_test,
                                                                 target_test)

                    training_errors.append(train_error)
                    validation_errors.append(validation_error)

                    epoch_finish_time = time.time()
                    training.epoch_time = epoch_finish_time - epoch_start_time

                    if epoch % training.show_epoch == 0 or is_first_iteration:
                        summary.show_last()
                        last_epoch_shown = epoch

                    if epoch_end_signal is not None:
                        epoch_end_signal(self)

                    is_first_iteration = False

                except StopTraining as err:
                    # TODO: This notification breaks table view in terminal.
                    # I need to show it in a different way.
                    logs.message("TRAIN", "Epoch #{} stopped. {}"
                                          "".format(epoch, str(err)))
                    break

            if epoch != last_epoch_shown:
                summary.show_last()

            if train_end_signal is not None:
                train_end_signal(self)

            summary.finish()
            logs.newline()
