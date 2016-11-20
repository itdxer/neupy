import numpy as np
from scipy.optimize import minimize_scalar

from neupy.utils import asfloat
from neupy.core.properties import BoundedProperty, ChoiceProperty
from .base import SingleStepConfigurable


__all__ = ('LinearSearch',)


class LinearSearch(SingleStepConfigurable):
    """
    Linear search is a step selection algorithm.

    Parameters
    ----------
    tol : float
        Tolerance for termination, default to ``0.1``.
        Can be any number greater that ``0``.

    maxiter : int
        Maximum number of interations. Works only for
        the ``brent`` method. Defaults to ``10``.

    search_method : {{``gloden``, ``brent``}}
        Linear search method. Can be ``golden`` for
        golden search or ``brent`` for Brent's search,
        default to ``golden``.

    Warns
    -----
    {SingleStepConfigurable.Warns}

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
    ...     train_size=0.85
    ... )
    >>>
    >>> cgnet = algorithms.ConjugateGradient(
    ...     connection=[
    ...         layers.Input(13),
    ...         layers.Sigmoid(50),
    ...         layers.Sigmoid(1),
    ...     ],
    ...     search_method='golden',
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
    0.20752676697596578

    See Also
    --------
    :network:`ConjugateGradient`
    """
    tol = BoundedProperty(default=0.1, minval=0)
    maxiter = BoundedProperty(default=10, minval=1)
    search_method = ChoiceProperty(choices=['golden', 'brent'],
                                   default='golden')

    def train_epoch(self, input_train, target_train):
        train_epoch = self.methods.train_epoch
        prediction_error = self.methods.prediction_error

        params = [param for param, _ in self.init_train_updates()]
        param_defaults = [param.get_value() for param in params]

        def setup_new_step(new_step):
            for param_default, param in zip(param_defaults, params):
                param.set_value(param_default)

            self.variables.step.set_value(asfloat(new_step))
            train_epoch(input_train, target_train)
            # Train epoch returns neural network error that was before
            # training epoch step, that's why we need to compute
            # it second time.
            error = prediction_error(input_train, target_train)

            return np.where(np.isnan(error), np.inf, error)

        options = {'xtol': self.tol}
        if self.search_method == 'brent':
            options['maxiter'] = self.maxiter

        res = minimize_scalar(
            setup_new_step,
            tol=self.tol,
            method=self.search_method,
            options=options,
        )

        return setup_new_step(res.x)
