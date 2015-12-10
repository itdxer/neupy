import theano
import theano.tensor as T
from theano.ifelse import ifelse
from scipy.optimize import minimize_scalar

from neupy.core.properties import NonNegativeNumberProperty, ChoiceProperty
from .base import LearningRateConfigurable


__all__ = ('LinearSearch',)


def golden_search(f, x0, direction, minstep=1e-5, maxstep=50.,
                  maxiter=1024, tol=1e-5):

    def interval_location(prev_func_output, step, x0, direction, maxstep):
        func_output = f(x0 + step * direction)
        output_decrease = T.gt(prev_func_output, func_output)
        step = ifelse(output_decrease,
                      T.minimum(2. * step, maxstep),
                      step)

        stoprule = theano.scan_module.until(
            T.or_(
                T.lt(prev_func_output, func_output),
                step > maxstep
            )
        )
        return [func_output, step], stoprule

    def interval_reduction(a, b, c ,d, tol):
        fc = f(x0 + c * direction)
        fd = f(x0 + d * direction)

        a, b, c, d = ifelse(
            T.lt(fc, fd),
            [a, d, d - golden_ratio * (d - a), c],
            [c, b, d, c + golden_ratio * (b - c)]
        )

        stoprule = theano.scan_module.until(
            T.lt(T.abs_(c - d), tol)
        )

        return [a, b, c, d], stoprule

    (_, steps), _ = theano.scan(
        interval_location,
        outputs_info=[T.constant(asfloat(np.inf)),
                      T.constant(asfloat(minstep))],
        non_sequences=[x0, direction, maxstep],
        n_steps=maxiter
    )
    compute_maxstep = theano.function([], steps[-1])
    maxstep = compute_maxstep()
    print(maxstep)
    golden_ratio = asfloat((math.sqrt(5) - 1) / 2)

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
    compute_step = theano.function([], (a[-1] + b[-1]) / 2)
    return compute_step()


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
