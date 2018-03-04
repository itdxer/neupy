import numpy as np
import tensorflow as tf

from neupy.utils import asfloat, flatten
from neupy.core.properties import ChoiceProperty
from neupy.algorithms.gd import NoMultipleStepSelection
from neupy.algorithms.utils import parameter_values, setup_parameter_updates
from neupy.layers.utils import count_parameters
from .base import GradientDescent


__all__ = ('ConjugateGradient',)


def fletcher_reeves(gradient_old, gradient_new, weight_old_delta):
    return (
        tf.tensordot(gradient_new, gradient_new, 1) /
        tf.tensordot(gradient_old, gradient_old, 1)
    )


def polak_ribiere(gradient_old, gradient_new, weight_old_delta):
    return (
        tf.tensordot(gradient_new, gradient_new - gradient_old, 1) /
        tf.tensordot(gradient_old, gradient_old, 1)
    )


def hentenes_stiefel(gradient_old, gradient_new, weight_old_delta):
    gradient_delta = gradient_new - gradient_old
    return (
        tf.tensordot(gradient_delta, gradient_new, 1) /
        tf.tensordot(weight_old_delta, gradient_delta, 1)
    )


def conjugate_descent(gradient_old, gradient_new, weight_old_delta):
    return (
        -tf.norm(gradient_new) /
        tf.tensordot(weight_old_delta, gradient_old, 1)
    )


def liu_storey(gradient_old, gradient_new, weight_old_delta):
    return (
        tf.tensordot(gradient_new, gradient_new - gradient_old, 1) /
        tf.tensordot(weight_old_delta, gradient_old, 1)
    )


def dai_yuan(gradient_old, gradient_new, weight_old_delta):
    return (
        tf.tensordot(gradient_new, gradient_new, 1) /
        tf.tensordot(gradient_new - gradient_old, weight_old_delta, 1)
    )


class ConjugateGradient(NoMultipleStepSelection, GradientDescent):
    """
    Conjugate Gradient algorithm.

    Parameters
    ----------
    update_function : {{``fletcher_reeves``, ``polak_ribiere``,\
    ``hentenes_stiefel``, ``conjugate_descent``, ``liu_storey``,\
    ``dai_yuan``}}
        Update function. Defaults to ``fletcher_reeves``.

    {GradientDescent.Parameters}

    Attributes
    ----------
    {GradientDescent.Attributes}

    Methods
    -------
    {GradientDescent.Methods}

    Examples
    --------
    >>> from sklearn import datasets, preprocessing
    >>> from sklearn.model_selection import train_test_split
    >>> from neupy import algorithms, layers, estimators, environment
    >>>
    >>> environment.reproducible()
    >>>
    >>> dataset = datasets.load_boston()
    >>> data, target = dataset.data, dataset.target
    >>>
    >>> data_scaler = preprocessing.MinMaxScaler()
    >>> target_scaler = preprocessing.MinMaxScaler()
    >>>
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...     data_scaler.fit_transform(data),
    ...     target_scaler.fit_transform(target),
    ...     test_size=0.15
    ... )
    >>>
    >>> cgnet = algorithms.ConjugateGradient(
    ...     connection=[
    ...         layers.Input(13),
    ...         layers.Sigmoid(50),
    ...         layers.Sigmoid(1),
    ...     ],
    ...     search_method='golden',
    ...     update_function='fletcher_reeves',
    ...     addons=[algorithms.LinearSearch],
    ...     verbose=False
    ... )
    >>>
    >>> cgnet.train(x_train, y_train, epochs=100)
    >>> y_predict = cgnet.predict(x_test).round(1)
    >>>
    >>> real = target_scaler.inverse_transform(y_test)
    >>> predicted = target_scaler.inverse_transform(y_predict)
    >>>
    >>> error = estimators.rmsle(real, predicted)
    >>> error
    0.2472330191179734

    See Also
    --------
    :network:`GradientDescent`: GradientDescent algorithm.
    :network:`LinearSearch`: Linear Search important algorithm for step \
    selection in Conjugate Gradient algorithm.
    """
    update_function = ChoiceProperty(
        default='fletcher_reeves',
        choices={
            'fletcher_reeves': fletcher_reeves,
            'polak_ribiere': polak_ribiere,
            'hentenes_stiefel': hentenes_stiefel,
            'conjugate_descent': conjugate_descent,
            'liu_storey': liu_storey,
            'dai_yuan': dai_yuan,
        }
    )

    def init_variables(self):
        super(ConjugateGradient, self).init_variables()
        n_parameters = count_parameters(self.connection)

        self.variables.update(
            prev_delta=tf.Variable(
                tf.zeros([n_parameters]),
                name="conj-grad/prev-delta",
                dtype=tf.float32,
            ),
            prev_gradient=tf.Variable(
                tf.zeros([n_parameters]),
                name="conj-grad/prev-gradient",
                dtype=tf.float32,
            ),
        )

    def init_train_updates(self):
        step = self.variables.step
        previous_delta = self.variables.prev_delta
        previous_gradient = self.variables.prev_gradient

        n_parameters = count_parameters(self.connection)
        parameters = parameter_values(self.connection)
        param_vector = tf.concat([flatten(param) for param in parameters],
                                 axis=0)

        gradients = tf.gradients(self.variables.error_func, parameters)
        full_gradient = tf.concat([flatten(grad) for grad in gradients],
                                  axis=0)

        beta = self.update_function(
            previous_gradient, full_gradient, previous_delta)

        parameter_delta = tf.where(
            tf.equal(tf.mod(self.variables.epoch, n_parameters), 1),
            -full_gradient,
            -full_gradient + beta * previous_delta
        )

        updated_parameters = param_vector + step * parameter_delta
        parameter_updates = setup_parameter_updates(
            parameters, updated_parameters)

        return parameter_updates + [
            (previous_gradient, full_gradient),
            (previous_delta, parameter_delta),
        ]
