from scipy.optimize import minimize_scalar

from neupy.core.properties import NonNegativeNumberProperty, ChoiceProperty
from .base import SingleStep


__all__ = ('LinearSearch',)


class LinearSearch(SingleStep):
    """ Linear search for the step selection. Basicly this algorithms
    try different steps and compute your predicted error, after few
    iteration it will chose one which was better.

    Parameters
    ----------
    tol : float
        Tolerance for termination, default to ``0.3``. Can be any number
        greater that zero.
    search_method : 'gloden', 'brent'
        Linear search method. Can be ``golden`` for golden search or ``brent``
        for Brent's search, default to ``golden``.

    Attributes
    ----------
    {first_step}

    Warns
    -----
    {bp_depending}

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
    :network:`ConjugateGradient`
    """
    tol = NonNegativeNumberProperty(default=0.3)
    search_method = ChoiceProperty(choices=['golden', 'brent'],
                                   default='golden')

    def set_weights(self, new_weights):
        for layer, new_weight in zip(self.train_layers, new_weights):
            layer.weight = new_weight.copy()

    def check_updates(self, new_step, weights, delta):
        self.set_weights(weights)
        self.step = new_step

        super(LinearSearch, self).update_weights(delta)
        predicted_output = self.predict(self.input_train)
        return self.error(predicted_output, self.target_train)

    def update_weights(self, weight_deltas):
        real_weights = [layer.weight for layer in self.train_layers]
        res = minimize_scalar(
            self.check_updates, args=(real_weights, weight_deltas),
            tol=self.tol, method=self.search_method,
            options={'xtol': self.tol}
        )

        self.set_weights(real_weights)
        self.step = res.x

        return super(LinearSearch, self).update_weights(weight_deltas)
