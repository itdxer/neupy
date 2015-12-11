import math

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

from neupy.utils import asfloat
from neupy.core.properties import (BoundedProperty, ChoiceProperty,
                                   NonNegativeIntProperty)
from .base import LearningRateConfigurable


__all__ = ('LinearSearch',)


def interval_location(f, minstep=1e-5, maxstep=50., maxiter=1024):
    """ Identify interval where potentialy could be optimal step.

    Parameters
    ----------
    f : func
    minstep : float
        Defaults to ``1e-5``.
    maxstep : float
        Defaults to ``50``.
    maxiter : int
        Defaults to ``1024``.
    tol : float
        Defaults to ``1e-5``.

    Returns
    -------
    float
        Right bound of interval where could be optimal step in
        specified direction. In case if there is no such direction
        function return ``maxstep`` instead.
    """

    def find_right_bound(prev_func_output, step, maxstep):
        func_output = f(step)
        is_output_decrease = T.gt(prev_func_output, func_output)
        step = ifelse(
            is_output_decrease,
            T.minimum(2. * step, maxstep),
            step
        )

        is_output_increse = T.lt(prev_func_output, func_output)
        stoprule = theano.scan_module.until(
            T.or_(is_output_increse, step > maxstep)
        )
        return [func_output, step], stoprule

    (_, steps), _ = theano.scan(
        find_right_bound,
        outputs_info=[T.constant(asfloat(np.inf)),
                      T.constant(asfloat(minstep))],
        non_sequences=[maxstep],
        n_steps=maxiter
    )
    find_maxstep = theano.function([], steps[-1])
    return find_maxstep()


def golden_search(f, maxstep=50, maxiter=1024, tol=1e-5):
    """ Identify best step for function in specific direction.

    Parameters
    ----------
    f : func
    maxstep : float
        Defaults to ``50``.
    maxiter : int
        Defaults to ``1024``.
    tol : float
        Defaults to ``1e-5``.

    Returns
    -------
    float
        Identified optimal step.
    """

    golden_ratio = asfloat((math.sqrt(5) - 1) / 2)

    def interval_reduction(a, b, c, d, tol):
        fc = f(c)
        fd = f(d)

        a, b, c, d = ifelse(
            T.lt(fc, fd),
            [a, d, d - golden_ratio * (d - a), c],
            [c, b, d, c + golden_ratio * (b - c)]
        )

        stoprule = theano.scan_module.until(
            T.lt(T.abs_(c - d), tol)
        )
        return [a, b, c, d], stoprule

    a = T.constant(asfloat(0))
    b = T.constant(asfloat(maxstep))
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)

    (a, b, _, _), _ = theano.scan(
        interval_reduction,
        outputs_info=[a, b, c, d],
        non_sequences=[asfloat(tol)],
        n_steps=maxiter
    )
    find_best_step = theano.function([], (a[-1] + b[-1]) / 2)
    return find_best_step()


def fmin_golden_search(f, minstep=1e-5, maxstep=50., maxiter=1024, tol=1e-5):
    """ Minimize scalar function using Golden Search.

    Parameters
    ----------
    f : func
    minstep : float
        Defaults to ``1e-5``.
    maxstep : float
        Defaults to ``50``.
    maxiter : int
        Defaults to ``1024``.
    tol : float
        Defaults to ``1e-5``.

    Returns
    -------
    int
    """

    params = (
        ('maxiter', maxiter),
        ('minstep', minstep),
        ('maxstep', maxstep),
        ('tol', tol),
    )
    for param_name, param_value in params:
        if param_value <= 0:
            raise ValueError("Parameter `{}` should be greater than zero."
                             "".format(param_name))

    if minstep >= maxstep:
        raise ValueError("`minstep` should be smaller than `maxstep`")

    maxstep = interval_location(f, minstep, maxstep, maxiter)
    best_step = golden_search(f, maxstep, maxiter, tol)

    return best_step


class LinearSearch(LearningRateConfigurable):
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
    ...         layers.Sigmoid(13),
    ...         layers.Sigmoid(50),
    ...         layers.Output(1),
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

    tol = BoundedProperty(default=0.3, minsize=0)
    maxstep = BoundedProperty(default=50, minsize=0)
    maxiter = NonNegativeIntProperty(default=1024)
    search_method = ChoiceProperty(choices={'golden': fmin_golden_search},
                                   default='golden')

    # def set_weights(self, new_weights):
    #     for layer, new_weight in zip(self.train_layers, new_weights):
    #         layer.weight = new_weight.copy()
    #
    # def check_updates(self, new_step, weights, delta):
    #     self.set_weights(weights)
    #     self.step = new_step
    #
    #     super(LinearSearch, self).update_weights(delta)
    #     predicted_output = self.predict(self.input_train)
    #     return self.error(predicted_output, self.target_train)
    #
    # def update_weights(self, weight_deltas):
    #     real_weights = [layer.weight for layer in self.train_layers]
    #     res = minimize_scalar(
    #         self.check_updates, args=(real_weights, weight_deltas),
    #         tol=self.tol, method=self.search_method,
    #         options={'xtol': self.tol}
    #     )
    #
    #     self.set_weights(real_weights)
    #     self.step = res.x
    #
    #     return super(LinearSearch, self).update_weights(weight_deltas)
