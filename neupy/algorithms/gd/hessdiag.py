from __future__ import division

import theano.tensor as T

from neupy.core.properties import ProperFractionProperty
from .base import GradientDescent


__all__ = ('HessianDiagonal',)


class HessianDiagonal(GradientDescent):
    """ Hissian diagonal is a Hessian algorithm approximation which require
    only computation of hessian matrix diagonal elements and makes it
    invertion much easier and faster.

    Parameters
    ----------
    min_eigval : float
        Setup min eigenvalue for Hessian diagonale matrix. After few
        iteration elements would be extremly small and matrix inverse
        produce huge number in hessian diagonal elements. This
        parameter control diagonal elements size. Defaults to ``1e-2``.
    {GradientDescent.optimizations}
    {ConstructableNetwork.connection}
    {SupervisedConstructableNetwork.error}
    {BaseNetwork.step}
    {BaseNetwork.show_epoch}
    {BaseNetwork.shuffle_data}
    {BaseNetwork.epoch_end_signal}
    {BaseNetwork.train_end_signal}
    {Verbose.verbose}

    Methods
    -------
    {supervised_train}
    {predict_raw}
    {full_methods}

    Examples
    --------
    Simple example

    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> hdnet = algorithms.HessianDiagonal(
    ...     (2, 3, 1),
    ...     verbose=False
    ... )
    >>> hdnet.train(x_train, y_train)

    Diabets dataset example

    >>> import numpy as np
    >>> from sklearn.cross_validation import train_test_split
    >>> from sklearn import datasets, preprocessing
    >>> from neupy import algorithms, layers
    >>> from neupy.functions import rmsle
    >>>
    >>> np.random.seed(0)
    >>>
    >>> dataset = datasets.load_diabetes()
    >>> data, target = dataset.data, dataset.target
    >>>
    >>> input_scaler = preprocessing.StandardScaler()
    >>> target_scaler = preprocessing.StandardScaler()
    >>>
    >>> x_train, x_test, y_train, y_test = train_test_split(
    ...     input_scaler.fit_transform(data),
    ...     target_scaler.fit_transform(target),
    ...     train_size=0.8
    ... )
    >>>
    >>> nw = algorithms.HessianDiagonal(
    ...     connection=[
    ...         layers.Sigmoid(10),
    ...         layers.Sigmoid(20),
    ...         layers.Output(1)
    ...     ],
    ...     step=1.5,
    ...     shuffle_data=False,
    ...     verbose=False,
    ...     min_eigval=1e-10
    ... )
    >>> nw.train(x_train, y_train, epochs=10)
    >>> y_predict = nw.predict(x_test)
    >>>
    >>> error = rmsle(target_scaler.inverse_transform(y_test),
    ...               target_scaler.inverse_transform(y_predict).round())
    >>> error
    0.50315919814691346

    See Also
    --------
    :network:`GradientDescent` : GradientDescent algorithm.
    :network:`Hessian` : Newton's method.
    """
    min_eigval = ProperFractionProperty(default=1e-2)

    def init_param_updates(self, layer, parameter):
        step = layer.step or self.variables.step
        min_eigval = self.min_eigval

        grad = T.grad(self.variables.error_func, wrt=parameter)
        hessian_diag = T.grad(grad.sum(), wrt=parameter)

        hessian_diag = T.switch(
            T.abs_(hessian_diag) < min_eigval,
            T.switch(
                hessian_diag < 0,
                -min_eigval,
                min_eigval,
            ),
            hessian_diag
        )
        return [
            # Divide gradient by hessian diagonal elementwise
            # mean that we just took diagonal hessian inverse (which is
            # reciprocal for each diagonal element) and mutliply
            # by gradient. This operation less clear by much faster.
            (parameter, parameter - step * grad / hessian_diag),
        ]
