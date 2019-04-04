from __future__ import division

import time

import numpy as np
import tensorflow as tf

from neupy import layers
from neupy.core.properties import (
    FunctionWithOptionsProperty,
    ScalarVariableProperty,
    IntProperty, Property,
)
from neupy.utils import (
    AttributeKeyDict, format_data,
    as_tuple, iters, tf_utils,
)
from neupy.algorithms.gd import objectives
from neupy.exceptions import InvalidConnection
from neupy.algorithms.base import BaseNetwork


__all__ = ('BaseOptimizer', 'GradientDescent')


class BaseOptimizer(BaseNetwork):
    """
    Gradient descent algorithm.

    Parameters
    ----------
    network : list, tuple or LayerConnection instance
        Network's architecture. There are a few ways
        to define it.

        - List of layers.
          For instance, ``[Input(2), Tanh(4), Relu(1)]``.

        - Constructed layers.
          For instance, ``Input(2) >> Tanh(4) >> Relu(1)``.

    regularizer : function or None
        Network's regularizer.

    loss : str or function
        Error/loss function. Defaults to ``mse``.

        - ``mae`` - Mean Absolute Error.

        - ``mse`` - Mean Squared Error.

        - ``rmse`` - Root Mean Squared Error.

        - ``msle`` - Mean Squared Logarithmic Error.

        - ``rmsle`` - Root Mean Squared Logarithmic Error.

        - ``categorical_crossentropy`` - Categorical cross entropy.

        - ``binary_crossentropy`` - Binary cross entropy.

        - ``binary_hinge`` - Binary hinge entropy.

        - ``categorical_hinge`` - Categorical hinge entropy.

        - Custom function which accepts two mandatory arguments.
          The first one is expected value and the second one is
          predicted value. Example:

        .. code-block:: python

            def custom_func(expected, predicted):
                return expected - predicted

    step : float, Variable
        Learning rate, defaults to ``0.1``.

    {BaseNetwork.show_epoch}

    {BaseNetwork.shuffle_data}

    {BaseNetwork.signals}

    {BaseNetwork.verbose}

    Attributes
    ----------
    {BaseNetwork.Attributes}

    Methods
    -------
    {BaseSkeleton.predict}

    train(X_train, y_train, X_test=None, y_test=None, epochs=100)
        Train network. You can control network's training procedure
        with ``epochs`` parameter. The ``X_test`` and ``y_test`` should
        be presented both in case network's validation required
        after each training epoch.

    {BaseSkeleton.fit}
    """
    step = ScalarVariableProperty(default=0.1)
    target = Property(default=None, allow_none=True)
    regularizer = Property(default=None, allow_none=True)
    loss = FunctionWithOptionsProperty(default='mse', choices={
        'mae': objectives.mae,
        'mse': objectives.mse,
        'rmse': objectives.rmse,
        'msle': objectives.msle,
        'rmsle': objectives.rmsle,

        'binary_crossentropy': objectives.binary_crossentropy,
        'categorical_crossentropy': objectives.categorical_crossentropy,

        'binary_hinge': objectives.binary_hinge,
        'categorical_hinge': objectives.categorical_hinge,
    })

    def __init__(self, network, options=None, **kwargs):
        options = options or kwargs

        if isinstance(network, (list, tuple)):
            network = layers.join(*network)

        self.network = network

        if len(self.network.output_layers) != 1:
            n_outputs = len(network.output_layers)

            raise InvalidConnection(
                "Connection should have one output "
                "layer, got {}".format(n_outputs))

        target = options.get('target')
        if target is not None and isinstance(target, (list, tuple)):
            options['target'] = tf.placeholder(tf.float32, shape=target)

        self.target = self.network.targets
        super(BaseOptimizer, self).__init__(**options)

        start_init_time = time.time()
        self.logs.message(
            "TENSORFLOW",
            "Initializing Tensorflow variables and functions.")

        self.variables = AttributeKeyDict()
        self.functions = AttributeKeyDict()
        self.network.outputs
        self.init_functions()

        self.logs.message(
            "TENSORFLOW",
            "Initialization finished successfully. It took {:.2f} seconds"
            "".format(time.time() - start_init_time))

    def init_train_updates(self):
        raise NotImplementedError()

    def init_functions(self):
        loss = self.loss(self.target, self.network.outputs)
        val_loss = self.loss(self.target, self.network.training_outputs)

        if self.regularizer is not None:
            loss += self.regularizer(self.network)

        self.variables.update(
            step=self.step,
            loss=loss,
            val_loss=val_loss,
        )

        with tf.name_scope('training-updates'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                training_updates = self.init_train_updates()
                training_updates.extend(update_ops)

        tf_utils.initialize_uninitialized_variables()

        with tf.name_scope('optimizer'):
            self.functions.update(
                predict=tf_utils.function(
                    inputs=as_tuple(self.network.inputs),
                    outputs=self.network.outputs,
                    name='predict'
                ),
                one_training_update=tf_utils.function(
                    inputs=as_tuple(self.network.inputs, self.target),
                    outputs=loss,
                    updates=training_updates,
                    name='one-update-step'
                ),
                score=tf_utils.function(
                    inputs=as_tuple(self.network.inputs, self.target),
                    outputs=val_loss,
                    name='score'
                ),
            )

    def format_input(self, X):
        X = as_tuple(X)
        X_formatted = []

        if len(X) != len(self.network.input_layers):
            raise ValueError(
                "Number of inputs doesn't match number "
                "of input layers in the network.")

        for input, input_layer in zip(X, self.network.input_layers):
            input_shape = tf.TensorShape(input_layer.input_shape)
            is_feature1d = (input_shape.ndims == 2 and input_shape[1] == 1)
            formatted_input = format_data(input, is_feature1d=is_feature1d)

            if (formatted_input.ndim + 1) == input_shape.ndims:
                # We assume that when one dimension was missed than user
                # wants to propagate single sample through the network
                formatted_input = np.expand_dims(formatted_input, axis=0)

            X_formatted.append(formatted_input)

        return X_formatted

    def format_target(self, y):
        output_shape = tf.TensorShape(self.network.output_shape)
        is_feature1d = (output_shape.ndims == 2 and output_shape[1] == 1)
        formatted_target = format_data(y, is_feature1d=is_feature1d)

        if (formatted_target.ndim + 1) == len(output_shape):
            # We assume that when one dimension was missed than user
            # wants to propagate single sample through the network
            formatted_target = np.expand_dims(formatted_target, axis=0)

        return formatted_target

    def score(self, X, y):
        """
        Calculate prediction accuracy for input data.

        Parameters
        ----------
        X : array-like
        y : array-like

        Returns
        -------
        float
            Prediction error.
        """
        X = self.format_input(X)
        y = self.format_target(y)
        return self.functions.score(*as_tuple(X, y))

    def predict(self, *X, **kwargs):
        """
        Makes a raw prediction.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        array-like
        """
        default_batch_size = getattr(self, 'batch_size', None)
        predict_kwargs = dict(
            batch_size=kwargs.pop('batch_size', default_batch_size),
            verbose=self.verbose,
        )

        # We require do to this check for python 2 compatibility
        if kwargs:
            raise TypeError("Unknown arguments: {}".format(kwargs))

        return self.network.predict(*self.format_input(X), **predict_kwargs)

    def train(self, X_train, y_train, X_test=None, y_test=None,
              *args, **kwargs):

        is_test_data_partialy_missing = (
            (X_test is None and y_test is not None) or
            (X_test is not None and y_test is None)
        )

        if is_test_data_partialy_missing:
            raise ValueError(
                "Input or target test samples are missed. They "
                "must be defined together or none of them.")

        X_train = self.format_input(X_train)
        y_train = self.format_target(y_train)

        if X_test is not None:
            X_test = self.format_input(X_test)
            y_test = self.format_target(y_test)

        return super(BaseOptimizer, self).train(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            *args, **kwargs)

    def one_training_update(self, X_train, y_train):
        return self.functions.one_training_update(
            *as_tuple(X_train, y_train))

    def get_params(self, deep=False, with_network=True):
        params = super(BaseOptimizer, self).get_params()
        if with_network:
            params['network'] = self.network
        return params

    def __reduce__(self):
        parameters = self.get_params(with_network=False)

        # We only need to know placeholders shape
        # in order to be able to reconstruct it
        parameters['target'] = tf_utils.shape_to_tuple(
            parameters['target'].shape)

        args = (self.network, parameters)
        return (self.__class__, args)

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__,
            self.network,
            self.repr_options())


class GradientDescent(BaseOptimizer):
    """
    Mini-batch Gradient Descent algorithm.

    Parameters
    ----------
    batch_size : int or None
        Set up min-batch size. The ``None`` value will ensure that all data
        samples will be propagated through the network at once.
        Defaults to ``128``.

    {BaseOptimizer.Parameters}

    Attributes
    ----------
    {BaseOptimizer.Attributes}

    Methods
    -------
    {BaseOptimizer.Methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>> from neupy.algorithms import *
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> network = Input(2) >> Sigmoid(3) >> Sigmoid(1)
    >>> optimizer = algorithms.GradientDescent(network, batch_size=1)
    >>> optimizer.train(x_train, y_train)
    """
    batch_size = IntProperty(default=128, minval=0, allow_none=True)

    def init_train_updates(self):
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.step,
        )
        self.functions.optimizer = optimizer
        return [optimizer.minimize(self.variables.loss)]

    def one_training_update(self, X_train, y_train):
        """
        Train one epoch.

        Parameters
        ----------
        X_train : array-like
            Training input dataset.

        y_train : array-like
            Training target dataset.

        Returns
        -------
        float
            Training error.
        """
        return self.functions.one_training_update(
            *as_tuple(X_train, y_train))

    def score(self, X, y):
        """
        Check the prediction error for the specified input samples
        and their targets.

        Parameters
        ----------
        X : array-like
        y : array-like

        Returns
        -------
        float
            Prediction error.
        """
        X = self.format_input(X)
        y = self.format_target(y)

        return iters.apply_batches(
            function=self.functions.score,
            inputs=as_tuple(X, y),
            batch_size=self.batch_size,
            show_output=True,
            show_progressbar=self.logs.enable,
            average_outputs=True,
        )
