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
    AttributeKeyDict, format_data, as_tuple, function,
    initialize_uninitialized_variables, iters,
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

    error : str or function
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
    regularizer = Property(default=None, allow_none=True)
    error = FunctionWithOptionsProperty(default='mse', choices={
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

        super(BaseOptimizer, self).__init__(**options)

        self.logs.message(
            "TENSORFLOW",
            "Initializing Tensorflow variables and functions."
        )
        start_init_time = time.time()

        self.variables = AttributeKeyDict()
        self.methods = AttributeKeyDict()

        self.init_input_output_variables()
        self.init_variables()
        self.init_methods()

        finish_init_time = time.time()
        self.logs.message(
            "TENSORFLOW",
            "Initialization finished successfully. It took {:.2f} seconds"
            "".format(finish_init_time - start_init_time))

    def iter_params_and_grads(self):
        layers, parameters = [], []

        for layer, _, parameter in self.network.iter_variables():
            layers.append(layer)
            parameters.append(parameter)

        gradients = tf.gradients(self.variables.error_func, parameters)

        for layer, parameter, gradient in zip(layers, parameters, gradients):
            yield layer, parameter, gradient

    def init_train_updates(self):
        raise NotImplementedError()

    def init_input_output_variables(self):
        output_layer = self.network.output_layers[0]
        self.variables.update(
            network_inputs=self.network.inputs,
            network_output=tf.placeholder(
                tf.float32,
                name='placeholder/target-{}'.format(output_layer.name),
            ),
        )

    def init_variables(self):
        network_output = self.variables.network_output
        loss = self.error(network_output, self.network.outputs)
        val_loss = self.error(network_output, self.network.training_outputs)

        if self.regularizer is not None:
            loss = loss + self.regularizer(self.network)

        self.variables.update(
            step=self.step,
            error_func=loss,
            validation_error_func=val_loss,
        )

    def init_methods(self):
        network_inputs = self.variables.network_inputs
        network_output = self.variables.network_output

        with tf.name_scope('training-updates'):
            training_updates = self.init_train_updates()

            for layer in self.network.layers:
                training_updates.extend(layer.updates)

            for variable in self.variables.values():
                if hasattr(variable, 'updates'):
                    training_updates.extend(variable.updates)

        initialize_uninitialized_variables()

        self.methods.update(
            predict=function(
                inputs=network_inputs,
                outputs=self.network.outputs,
                name='network/func-predict'
            ),
            one_training_update=function(
                inputs=as_tuple(network_inputs, network_output),
                outputs=self.variables.error_func,
                updates=training_updates,
                name='network/func-train-epoch'
            ),
            score=function(
                inputs=as_tuple(network_inputs, network_output),
                outputs=self.variables.validation_error_func,
                name='network/func-prediction-error'
            )
        )

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
        X_and_y = [format_data(x) for x in as_tuple(X, y)]
        return self.methods.score(*X_and_y)

    def predict(self, X):
        """
        Return prediction results for the input data.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        array-like
        """
        X = [format_data(x) for x in as_tuple(X)]
        return self.network.predict(*X)

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

        X_train = [format_data(x) for x in as_tuple(X_train)]
        y_train = format_data(y_train)

        if X_test is not None:
            X_test = [format_data(x) for x in as_tuple(X_test)]
            y_test = format_data(y_test)

        return super(BaseOptimizer, self).train(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            *args, **kwargs)

    def one_training_update(self, X_train, y_train):
        return self.methods.one_training_update(*as_tuple(X_train, y_train))

    def get_params(self, deep=False, with_network=True):
        params = super(BaseOptimizer, self).get_params()
        if with_network:
            params['network'] = self.network
        return params

    def __reduce__(self):
        parameters = self.get_params(with_network=False)
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
    >>> network = Input(2) > Sigmoid(3) > Sigmoid(1)
    >>> mgdnet = algorithms.GradientDescent(network, batch_size=1)
    >>> mgdnet.train(x_train, y_train)
    """
    batch_size = IntProperty(default=128, minval=0, allow_none=True)

    def init_train_updates(self):
        updates = []
        step = self.variables.step

        for _, parameter, gradient in self.iter_params_and_grads():
            updates.append((parameter, parameter - step * gradient))

        return updates

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
        return self.methods.one_training_update(*as_tuple(X_train, y_train))

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
        X = [format_data(x) for x in as_tuple(X)]
        y = format_data(y)

        return iters.apply_batches(
            function=self.methods.score,
            inputs=as_tuple(X, y),
            batch_size=self.batch_size,
            show_output=True,
            show_progressbar=self.logs.enable,
            average_outputs=True,
        )

    def predict(self, X):
        """
        Makes a raw prediction.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        array-like
        """
        outputs = iters.apply_batches(
            function=self.methods.predict,
            inputs=[format_data(x) for x in as_tuple(X)],
            batch_size=self.batch_size,
            show_progressbar=self.logs.enable,
        )
        return np.concatenate(outputs, axis=0)
