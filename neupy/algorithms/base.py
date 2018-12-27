# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, unicode_literals

import time
import types
import inspect
from abc import abstractmethod
from collections import defaultdict

import numpy as np

from neupy.exceptions import StopTraining
from neupy.core.logs import Verbose
from neupy.core.config import ConfigurableABC
from neupy.core.properties import Property, NumberProperty, IntProperty
from neupy.algorithms import signals as base_signals
from neupy.utils import iters, as_tuple


__all__ = ('BaseSkeleton', 'BaseNetwork')


def preformat_value(value):
    if inspect.isfunction(value) or inspect.isclass(value):
        return value.__name__

    elif isinstance(value, (list, tuple, set)):
        return [preformat_value(v) for v in value]

    elif isinstance(value, (np.ndarray, np.matrix)):
        return value.shape

    elif hasattr(value, 'default'):
        return value.default

    return value


class BaseSkeleton(ConfigurableABC, Verbose):
    """
    Base class for neural network algorithms.

    Methods
    -------
    fit(\*args, \*\*kwargs)
        Alias to the ``train`` method.

    predict(X)
        Predicts output for the specified input.
    """

    def __init__(self, *args, **options):
        super(BaseSkeleton, self).__init__(*args, **options)

        self.logs.title("Main information")
        self.logs.message("ALGORITHM", self.__class__.__name__)
        self.logs.newline()

        for key, data in sorted(self.options.items()):
            formated_value = preformat_value(getattr(self, key))
            msg_text = "{} = {}".format(key, formated_value)
            self.logs.message("OPTION", msg_text, color='green')

        self.logs.newline()

    @abstractmethod
    def train(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError()

    def transform(self, X):
        return self.predict(X)

    def fit(self, X, y=None, *args, **kwargs):
        self.train(X, y, *args, **kwargs)
        return self

    def repr_options(self):
        options = []
        for option_name in self.options:
            option_value = getattr(self, option_name)
            option_value = preformat_value(option_value)

            option_repr = "{}={}".format(option_name, option_value)
            options.append(option_repr)

        return ', '.join(options)

    def __repr__(self):
        class_name = self.__class__.__name__
        available_options = self.repr_options()
        return "{}({})".format(class_name, available_options)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


class Events(object):
    def __init__(self, network, signals):
        self.data = defaultdict(list)
        self.network = network
        self.signals = signals

    def trigger(self, name, **data):
        if data:
            self.data[name].append(data)

        for signal in self.signals:
            if hasattr(signal, name):
                signal_method = getattr(signal, name)
                signal_method(self.network, **data)


class BaseNetwork(BaseSkeleton):
    """
    Base class for Neural Network algorithms.

    Parameters
    ----------
    step : float
        Learning rate, defaults to ``0.1``.

    show_epoch : int
        This property controls how often the network will display
        information about training. It has to be defined as positive
        integer. For instance, number ``100`` mean that network shows
        summary at 1st, 100th, 200th, 300th ... and last epochs.

        Defaults to ``1``.

    shuffle_data : bool
        If it's ``True`` than training data will be shuffled before
        the training. Defaults to ``True``.

    signals : dict, list or function
        Function that will be triggered after certain events during
        the training.

    {Verbose.Parameters}

    Attributes
    ----------
    training_errors : list
        List of the training errors.

    validation_errors : list
        Contains list of validation errors. Validation error will be equal
        to ``None`` when validation wasn't done during this epoch.

    last_epoch : int
        Value equals to the last trained epoch. After initialization
        it is equal to ``0``.
    """
    step = NumberProperty(default=0.1, minval=0)
    show_epoch = IntProperty(minval=1, default=1)
    shuffle_data = Property(default=False, expected_type=bool)
    signals = Property(expected_type=object)

    def __init__(self, *args, **options):
        super(BaseNetwork, self).__init__(*args, **options)

        self.last_epoch = 0
        self.n_updates_made = 0

        signals = list(as_tuple(
            self.signals,
            base_signals.PrintLastErrorSignal(),
            base_signals.ProgressbarSignal(),
        ))

        for i, signal in enumerate(signals):
            if isinstance(signal, (types.FunctionType, types.LambdaType)):
                signals[i] = base_signals.EpochEndSignal(signal)

        self.events = Events(network=self, signals=signals)

    def predict(self, X):
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

    def one_training_update(self, X_train, y_train=None):
        """
        Function would be trigger before run all training procedure
        related to the current epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        raise NotImplementedError()

    def score(self, X_test, y_test):
        raise NotImplementedError()

    @property
    def training_errors(self):
        errors = self.events.data.get('train_error', [])
        return [error['value'] for error in errors]

    @property
    def validation_errors(self):
        errors = self.events.data.get('valid_error', [])
        return [error['value'] for error in errors]

    def train(self, X_train, y_train=None, X_test=None, y_test=None,
              epochs=100, batch_size=None):
        """
        Method train neural network.

        epochs : int
            Defaults to `100`.

        epsilon : float or None
            Defaults to ``None``.
        """
        if epochs <= 0:
            raise ValueError("Number of epochs needs to be a positive number")

        epochs = int(epochs)
        first_epoch = self.last_epoch + 1
        batch_size = batch_size or getattr(self, 'batch_size', None)

        self.events.trigger(
            name='train_start',
            X_train=X_train,
            y_train=y_train,
            epochs=epochs,
            batch_size=batch_size
        )

        try:
            for epoch in range(first_epoch, first_epoch + epochs):
                self.events.trigger('epoch_start')

                self.last_epoch = epoch
                iterator = iters.minibatches(
                    (X_train, y_train),
                    batch_size,
                    self.shuffle_data,
                )

                for X_batch, y_batch in iterator:
                    self.events.trigger('update_start')
                    update_start_time = time.time()

                    train_error = self.one_training_update(X_batch, y_batch)
                    self.n_updates_made += 1

                    self.events.trigger(
                        name='train_error',
                        value=train_error,
                        eta=time.time() - update_start_time,
                        epoch=epoch,
                        n_updates=self.n_updates_made - 1,
                    )
                    self.events.trigger('update_end')

                if X_test is not None:
                    test_start_time = time.time()
                    validation_error = self.score(X_test, y_test)
                    self.events.trigger(
                        name='valid_error',
                        value=validation_error,
                        eta=time.time() - test_start_time,
                        epoch=epoch,
                        n_updates=self.n_updates_made,
                    )

                self.events.trigger('epoch_end')

        except StopTraining as err:
            self.logs.message(
                "TRAIN",
                "Epoch #{} was stopped. Message: {}".format(epoch, str(err)))

        self.events.trigger('train_end')
