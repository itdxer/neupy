# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, unicode_literals

import time
import inspect
from abc import abstractmethod

import numpy as np

from neupy.exceptions import StopTraining
from neupy.core.logs import Verbose
from neupy.core.config import ConfigurableABC
from neupy.core.properties import Property, NumberProperty, IntProperty
from neupy.algorithms import signals as base_signals
from neupy.algorithms.plots import plot_optimizer_errors
from neupy.utils import iters, as_tuple


__all__ = ('BaseSkeleton', 'BaseNetwork')


def preformat_value(value):
    if inspect.isfunction(value) or inspect.isclass(value):
        return value.__name__

    elif isinstance(value, (list, tuple, set)):
        return [preformat_value(v) for v in value]

    elif isinstance(value, (np.ndarray, np.matrix)):
        return value.shape

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
        self.network = network
        self.signals = signals
        self.logs = []

    def trigger(self, name, store_data=False, **data):
        if store_data and data:
            self.logs.append(dict(data, name=name))

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

    Methods
    -------
    {BaseSkeleton.fit}

    predict(X)
        Propagates input ``X`` through the network and
        returns produced output.

    plot_errors(logx=False, show=True, **figkwargs)
        Using errors collected during the training this method
        generates plot that can give additional insight into the
        performance reached during the training.

    Attributes
    ----------
    errors : list
        Information about errors. It has two main attributes, namely
        ``train`` and ``valid``. These attributes provide access to
        the training and validation errors respectively.

    last_epoch : int
        Value equals to the last trained epoch. After initialization
        it is equal to ``0``.

    n_updates_made : int
        Number of training updates applied to the network.
    """
    step = NumberProperty(default=0.1, minval=0)
    show_epoch = IntProperty(minval=1, default=1)
    shuffle_data = Property(default=False, expected_type=bool)
    signals = Property(expected_type=object)

    def __init__(self, *args, **options):
        super(BaseNetwork, self).__init__(*args, **options)

        self.last_epoch = 0
        self.n_updates_made = 0
        self.errors = base_signals.ErrorCollector()

        signals = list(as_tuple(
            base_signals.ProgressbarSignal(),
            base_signals.PrintLastErrorSignal(),
            self.errors,
            self.signals,
        ))

        for i, signal in enumerate(signals):
            if inspect.isfunction(signal):
                signals[i] = base_signals.EpochEndSignal(signal)

            elif inspect.isclass(signal):
                signals[i] = signal()

        self.events = Events(network=self, signals=signals)

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

    def score(self, X, y):
        raise NotImplementedError()

    def plot_errors(self, logx=False, show=True, **figkwargs):
        return plot_optimizer_errors(
            optimizer=self,
            logx=logx,
            show=show,
            **figkwargs
        )

    def train(self, X_train, y_train=None, X_test=None, y_test=None,
              epochs=100, batch_size=None):
        """
        Method train neural network.

        Parameters
        ----------
        X_train : array-like
        y_train : array-like or None
        X_test : array-like or None
        y_test : array-like or None

        epochs : int
            Defaults to ``100``.

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
            batch_size=batch_size,
            store_data=False,
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
                        n_updates=self.n_updates_made,
                        n_samples=iters.count_samples(X_batch),
                        store_data=True,
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
                        n_samples=iters.count_samples(X_test),
                        store_data=True,
                    )

                self.events.trigger('epoch_end')

        except StopTraining as err:
            self.logs.message(
                "TRAIN",
                "Epoch #{} was stopped. Message: {}".format(epoch, str(err)))

        self.events.trigger('train_end')
