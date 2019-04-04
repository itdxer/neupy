import tensorflow as tf

from neupy.utils import dot, function_name_scope, asfloat, make_single_vector
from neupy.core.properties import (ChoiceProperty, NumberProperty,
                                   WithdrawProperty)
from neupy.utils.tf_utils import setup_parameter_updates
from .base import BaseOptimizer
from .quasi_newton import safe_division, WolfeLineSearchForStep


__all__ = ('ConjugateGradient',)


@function_name_scope
def fletcher_reeves(old_g, new_g, delta_w, epsilon=1e-7):
    return safe_division(
        dot(new_g, new_g),
        dot(old_g, old_g),
        epsilon,
    )


@function_name_scope
def polak_ribiere(old_g, new_g, delta_w, epsilon=1e-7):
    return safe_division(
        dot(new_g, new_g - old_g),
        dot(old_g, old_g),
        epsilon,
    )


@function_name_scope
def hentenes_stiefel(old_g, new_g, delta_w, epsilon=1e-7):
    gradient_delta = new_g - old_g
    return safe_division(
        dot(gradient_delta, new_g),
        dot(delta_w, gradient_delta),
        epsilon,
    )


@function_name_scope
def liu_storey(old_g, new_g, delta_w, epsilon=1e-7):
    return -safe_division(
        dot(new_g, new_g - old_g),
        dot(delta_w, old_g),
        epsilon,
    )


@function_name_scope
def dai_yuan(old_g, new_g, delta_w, epsilon=1e-7):
    return safe_division(
        dot(new_g, new_g),
        dot(new_g - old_g, delta_w),
        epsilon,
    )


class ConjugateGradient(WolfeLineSearchForStep, BaseOptimizer):

    """
    Conjugate Gradient algorithm.

    Parameters
    ----------
    update_function : ``fletcher_reeves``, ``polak_ribiere``,\
    ``hentenes_stiefel``, ``dai_yuan``, ``liu_storey``
        Update function. Defaults to ``fletcher_reeves``.

    epsilon : float
        Ensures computational stability during the division in
        ``update_function`` when denominator is very small number.
        Defaults to ``1e-7``.

    {WolfeLineSearchForStep.Parameters}

    {BaseOptimizer.network}

    {BaseOptimizer.loss}

    {BaseOptimizer.show_epoch}

    {BaseOptimizer.shuffle_data}

    {BaseOptimizer.signals}

    {BaseOptimizer.verbose}

    {BaseOptimizer.regularizer}

    Attributes
    ----------
    {BaseOptimizer.Attributes}

    Methods
    -------
    {BaseOptimizer.Methods}

    Examples
    --------
    >>> from sklearn import datasets, preprocessing
    >>> from sklearn.model_selection import train_test_split
    >>> from neupy import algorithms, layers
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
    ...     network=[
    ...         layers.Input(13),
    ...         layers.Sigmoid(50),
    ...         layers.Sigmoid(1),
    ...     ],
    ...     update_function='fletcher_reeves',
    ...     verbose=False
    ... )
    >>>
    >>> cgnet.train(x_train, y_train, epochs=100)
    >>> y_predict = cgnet.predict(x_test).round(1)
    >>>
    >>> real = target_scaler.inverse_transform(y_test)
    >>> predicted = target_scaler.inverse_transform(y_predict)

    References
    ----------
    [1] Jorge Nocedal, Stephen J. Wright, Numerical Optimization.
        Chapter 5, Conjugate Gradient Methods, p. 101-133
    """
    epsilon = NumberProperty(default=1e-7, minval=0)
    update_function = ChoiceProperty(
        default='fletcher_reeves',
        choices={
            'fletcher_reeves': fletcher_reeves,
            'polak_ribiere': polak_ribiere,
            'hentenes_stiefel': hentenes_stiefel,
            'liu_storey': liu_storey,
            'dai_yuan': dai_yuan,
        }
    )
    step = WithdrawProperty()

    def init_functions(self):
        n_parameters = self.network.n_parameters
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
            iteration=tf.Variable(
                asfloat(self.last_epoch),
                name='conj-grad/current-iteration',
                dtype=tf.float32
            ),
        )
        super(ConjugateGradient, self).init_functions()

    def init_train_updates(self):
        iteration = self.variables.iteration
        previous_delta = self.variables.prev_delta
        previous_gradient = self.variables.prev_gradient

        n_parameters = self.network.n_parameters
        variables = self.network.variables
        parameters = [var for var in variables.values() if var.trainable]
        param_vector = make_single_vector(parameters)

        gradients = tf.gradients(self.variables.loss, parameters)
        full_gradient = make_single_vector(gradients)

        beta = self.update_function(
            previous_gradient, full_gradient, previous_delta, self.epsilon)

        parameter_delta = tf.where(
            tf.equal(tf.mod(iteration, n_parameters), 0),
            -full_gradient,
            -full_gradient + beta * previous_delta
        )

        step = self.find_optimal_step(param_vector, parameter_delta)
        updated_parameters = param_vector + step * parameter_delta
        updates = setup_parameter_updates(parameters, updated_parameters)

        # We have to compute these values first, otherwise
        # parallelization, in tensorflow, can mix update order
        # and, for example, previous gradient can be equal to
        # current gradient value. It happens because tensorflow
        # try to execute operations in parallel.
        with tf.control_dependencies([full_gradient, parameter_delta]):
            updates.extend([
                previous_gradient.assign(full_gradient),
                previous_delta.assign(parameter_delta),
                iteration.assign(iteration + 1),
            ])

        return updates
