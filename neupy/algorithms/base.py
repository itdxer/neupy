# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, unicode_literals

import time
import types

from neupy.utils import preformat_value, as_tuple
from neupy.exceptions import StopTraining
from neupy.core.base import BaseSkeleton
from neupy.core.properties import Property, NumberProperty, IntProperty
from .utils import iter_until_converge, shuffle, format_time


__all__ = ('BaseNetwork',)


class BaseNetwork(BaseSkeleton):
    """
    Base class for Neural Network algorithms.

    Parameters
    ----------
    step : float
        Learning rate, defaults to ``0.1``.

    show_epoch : int
        This property controls how often the network will
        display information about training.

        It has to be defined as positive integer. It defines how offen
        would you like to see summary output in terminal. For instance,
        number ``100`` mean that network shows summary at 100th, 200th,
        300th ... epochs.

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
    training_errors : list
        List of the training errors.

    validation_errors : list
        Contains list of training errors. Validation error will be equal
        to ``None`` when validation wasn't done during this epoch.

    last_epoch : int
        Value equals to the last trained epoch. After initialization
        it is equal to ``0``.
    """
    step = NumberProperty(default=0.1, minval=0)

    show_epoch = IntProperty(minval=1, default=1)
    shuffle_data = Property(default=False, expected_type=bool)

    epoch_end_signal = Property(expected_type=types.FunctionType)
    train_end_signal = Property(expected_type=types.FunctionType)

    def __init__(self, *args, **options):
        self.training_errors = []
        self.validation_errors = []

        self.last_epoch = 0
        self.epoch_time = 0

        super(BaseNetwork, self).__init__(*args, **options)

        self.logs.title("Main information")
        self.logs.message("ALGORITHM", self.__class__.__name__)
        self.logs.newline()

        for key, data in sorted(self.options.items()):
            formated_value = preformat_value(getattr(self, key))
            msg_text = "{} = {}".format(key, formated_value)
            self.logs.message("OPTION", msg_text, color='green')

        self.logs.newline()

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
        raise NotImplementedError()

    def train_epoch(self, input_train, target_train=None):
        raise NotImplementedError()

    def score(self, input_test, target_test):
        raise NotImplementedError()

    def print_last_error(self):
        train_error = self.training_errors[-1]
        validation_error = self.validation_errors[-1]

        messages = []
        base_message = "#{} : [{}] ".format(
            self.last_epoch, format_time(self.epoch_time))

        if train_error is not None:
            messages.append("train: {:.6f}".format(train_error))

        if validation_error is not None:
            messages.append("valid: {:.6f}".format(validation_error))

        self.logs.write(base_message + ', '.join(messages))

    def train(self, input_train, target_train=None, input_test=None,
              target_test=None, epochs=100, epsilon=None):
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
        if epochs <= 0:
            raise ValueError("Number of epochs needs to be a positive number")

        last_epoch_shown = 0
        next_epoch = self.last_epoch + 1
        iterepochs = range(next_epoch, next_epoch + int(epochs))

        if epsilon is not None:
            iterepochs = iter_until_converge(self, epsilon, max_epochs=epochs)

        for epoch_index, epoch in enumerate(iterepochs):
            epoch_start_time = time.time()
            self.last_epoch = epoch

            if self.shuffle_data:
                data = shuffle(*as_tuple(input_train, target_train))
                input_train, target_train = data[:-1], data[-1]

                if len(input_train) == 1:
                    input_train = input_train[0]

            try:
                train_error = self.train_epoch(input_train, target_train)
                validation_error = None

                if input_test is not None:
                    validation_error = self.score(input_test, target_test)

                self.training_errors.append(train_error)
                self.validation_errors.append(validation_error)

                epoch_finish_time = time.time()
                self.epoch_time = epoch_finish_time - epoch_start_time

                if epoch % self.show_epoch == 0 or epoch_index == 0:
                    self.print_last_error()
                    last_epoch_shown = epoch

                if self.epoch_end_signal is not None:
                    self.epoch_end_signal(self)

            except StopTraining as err:
                self.logs.message(
                    "TRAIN", "Epoch #{} was stopped. Message: {}".format(
                        epoch, str(err))
                )
                break

        if epoch != last_epoch_shown:
            self.print_last_error()

        if self.train_end_signal is not None:
            self.train_end_signal(self)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
