import numpy as np
import theano
from theano.ifelse import ifelse
import theano.tensor as T

from neupy.utils import asfloat
from neupy.core.properties import ChoiceProperty
from .base import GradientDescent


__all__ = ('ConjugateGradient',)


def fletcher_reeves(gradient_old, gradient_new, weight_old_delta):
    return (
        T.dot(gradient_new, gradient_new) /
        T.dot(gradient_old, gradient_old)
    )


def polak_ribiere(gradient_old, gradient_new, weight_old_delta):
    return (
        T.dot(gradient_new, gradient_new - gradient_old) /
        T.dot(gradient_old, gradient_old)
    )


def hentenes_stiefel(gradient_old, gradient_new, weight_old_delta):
    gradient_delta = gradient_new - gradient_old
    return (
        T.dot(gradient_delta, gradient_new) /
        T.dot(weight_old_delta, gradient_delta)
    )


def conjugate_descent(gradient_old, gradient_new, weight_old_delta):
    return (
        -gradient_new.norm(L=2) /
        T.dot(weight_old_delta, gradient_old)
    )


def liu_storey(gradient_old, gradient_new, weight_old_delta):
    return (
        T.dot(gradient_new, gradient_new - gradient_old) /
        T.dot(weight_old_delta, gradient_old)
    )


def dai_yuan(gradient_old, gradient_new, weight_old_delta):
    return (
        T.dot(gradient_new, gradient_new) /
        T.dot(gradient_new - gradient_old, weight_old_delta)
    )


class ConjugateGradient(GradientDescent):
    """ Conjugate Gradient algorithm.

    Parameters
    ----------
    update_function : {{'fletcher_reeves', 'polak_ribiere',\
    'hentenes_stiefel', 'conjugate_descent', 'liu_storey', 'dai_yuan'}}
        Update function. Defaults to ``fletcher_reeves``.
    {optimizations}
    {full_params}

    Methods
    -------
    {supervised_train}
    {predict_raw}
    {full_methods}

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>>
    >>> from sklearn import datasets, preprocessing
    >>> from sklearn.cross_validation import train_test_split
    >>> from neupy import algorithms, layers
    >>> from neupy.functions import rmsle
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
    ...     train_size=0.85
    ... )
    >>>
    >>> cgnet = algorithms.ConjugateGradient(
    ...     connection=[
    ...         layers.Sigmoid(13),
    ...         layers.Sigmoid(50),
    ...         layers.Output(1),
    ...     ],
    ...     search_method='golden',
    ...     update_function='fletcher_reeves',
    ...     optimizations=[algorithms.LinearSearch],
    ...     verbose=False
    ... )
    >>>
    >>> cgnet.train(x_train, y_train, epochs=100)
    >>> y_predict = cgnet.predict(x_test)
    >>>
    >>> real = target_scaler.inverse_transform(y_test)
    >>> predicted = target_scaler.inverse_transform(y_predict)
    >>>
    >>> error = rmsle(real, predicted.round(1))
    >>> error
    0.20752676697596578

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

    def init_layers(self):
        super(ConjugateGradient, self).init_layers()
        for layer in self.train_layers:
            for parameter in layer.parameters:
                parameter_shape = T.shape(parameter).eval()
                parameter.prev_delta = theano.shared(
                    name="prev_delta_" + parameter.name,
                    value=asfloat(np.zeros(parameter_shape)),
                )
                parameter.prev_gradient = theano.shared(
                    name="prev_grad_" + parameter.name,
                    value=asfloat(np.zeros(parameter_shape)),
                )

    def init_param_updates(self, layer, parameter):
        step = layer.step or self.variables.step
        prev_gradient = parameter.prev_gradient
        prev_delta = parameter.prev_delta

        gradient = T.grad(self.variables.error_func, wrt=parameter)
        beta = self.update_function(
            prev_gradient.ravel(),
            gradient.ravel(),
            prev_delta
        )

        weight_delta = ifelse(
            T.eq(T.mod(self.variables.epoch, 25), 1),
            -gradient,
            -gradient + beta * prev_delta
        )

        return [
            (parameter, parameter + step * weight_delta),
            (prev_gradient, gradient),
            (prev_delta, weight_delta),
        ]
