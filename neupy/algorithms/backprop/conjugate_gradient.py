from operator import mul

from numpy import sqrt, inner

from neupy.core.properties import ChoiceProperty
from neupy.algorithms.utils import (matrix_list_in_one_vector,
                                    vector_to_list_of_matrix)
from .backpropagation import Backpropagation


__all__ = ('ConjugateGradient',)


def fletcher_reeves(gradient_old, gradient_new, weight_old_delta):
    return (
        inner(gradient_new, gradient_new) /
        inner(gradient_old, gradient_old)
    )


def polak_ribiere(gradient_old, gradient_new, weight_old_delta):
    return (
        inner(gradient_new, gradient_new - gradient_old) /
        inner(gradient_old, gradient_old)
    )


def hentenes_stiefel(gradient_old, gradient_new, weight_old_delta):
    gradient_delta = gradient_new - gradient_old
    return (
        inner(gradient_delta, gradient_new) /
        inner(weight_old_delta, gradient_delta)
    )


def conjugate_descent(gradient_old, gradient_new, weight_old_delta):
    # Note: `sqrt(dot(a.T, a))` works faster than `linalg.norm(a)`
    return (
        -sqrt(inner(gradient_new, gradient_new)) /
        inner(weight_old_delta, gradient_old)
    )


def liu_storey(gradient_old, gradient_new, weight_old_delta):
    return (
        inner(gradient_new, gradient_new - gradient_old) /
        inner(weight_old_delta, gradient_old)
    )


def dai_yuan(gradient_old, gradient_new, weight_old_delta):
    return (
        inner(gradient_new, gradient_new) /
        inner(gradient_new - gradient_old, weight_old_delta)
    )


class ConjugateGradient(Backpropagation):
    """ Conjugate Gradient algorithm.

    Parameters
    ----------
    update_function : {{'fletcher_reeves', 'polak_ribiere',\
    'hentenes_stiefel', 'conjugate_descent', 'liu_storey', 'dai_yuan'}}
        Update function. Defaults to ``fletcher_reeves``.
    {optimizations}
    {raw_predict_param}
    {full_params}

    Methods
    -------
    {supervised_train}
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
    ...         layers.SigmoidLayer(13),
    ...         layers.SigmoidLayer(50),
    ...         layers.OutputLayer(1),
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
    :network:`Backpropagation`: Backpropagation algorithm.
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
        self.n_weights = sum(mul(*layer.size) for layer in self.train_layers)

    def get_weight_delta(self, output_train, target_train):
        gradients = super(ConjugateGradient, self).get_gradient(output_train,
                                                                target_train)
        epoch = self.epoch
        gradient = matrix_list_in_one_vector(gradients)
        weight_delta = -gradient

        if epoch > 1 and epoch % self.n_weights == 0:
            # Must reset after every N iteration, because algoritm
            # lose conjugacy.
            self.logs.info("TRAIN", "Reset conjugate gradient vector")
            del self.prev_gradient

        if hasattr(self, 'prev_gradient'):
            gradient_old = self.prev_gradient
            weight_delta_old = self.prev_weight_delta
            beta = self.update_function(gradient_old, gradient,
                                        weight_delta_old)

            weight_delta += beta * weight_delta_old

        weight_deltas = vector_to_list_of_matrix(
            weight_delta,
            (layer.size for layer in self.train_layers)
        )

        self.prev_weight_delta = weight_delta.copy()
        self.prev_gradient = gradient.copy()

        return weight_deltas
