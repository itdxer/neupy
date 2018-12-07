from __future__ import division

import tensorflow as tf

from neupy.core.properties import ProperFractionProperty
from neupy.utils import flatten
from neupy.algorithms.gd import NoMultipleStepSelection
from neupy.algorithms.utils import (setup_parameter_updates, parameter_values,
                                    make_single_vector)
from .base import BaseGradientDescent


__all__ = ('HessianDiagonal',)


class HessianDiagonal(NoMultipleStepSelection, BaseGradientDescent):
    """
    Hissian diagonal is a Hessian algorithm approximation which require
    only computation of hessian matrix diagonal elements and makes it
    invertion much easier and faster.

    Parameters
    ----------
    min_eigval : float
        Set up minimum eigenvalue for Hessian diagonale matrix. After a few
        iteration elements will be extremly small and matrix inverse
        produce huge number in hessian diagonal elements. This
        parameter control diagonal elements size. Defaults to ``1e-2``.

    {BaseGradientDescent.Parameters}

    Attributes
    ----------
    {BaseGradientDescent.Attributes}

    Methods
    -------
    {BaseGradientDescent.Methods}

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> x_train = np.array([[1, 2], [3, 4]])
    >>> y_train = np.array([[1], [0]])
    >>>
    >>> hdnet = algorithms.HessianDiagonal((2, 3, 1))
    >>> hdnet.train(x_train, y_train)

    Diabets dataset example

    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn import datasets, preprocessing
    >>> from neupy import algorithms, layers, environment
    >>> from neupy.estimators import rmsle
    >>>
    >>> environment.reproducible()
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
    ...     test_size=0.2
    ... )
    >>>
    >>> nw = algorithms.HessianDiagonal(
    ...     connection=[
    ...         layers.Input(10),
    ...         layers.Sigmoid(20),
    ...         layers.Sigmoid(1),
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

    Notes
    -----
    - Method requires all training data during propagation, which means
      it's not allowed to use mini-batches.

    See Also
    --------
    :network:`BaseGradientDescent` : BaseGradientDescent algorithm.
    :network:`Hessian` : Newton's method.
    """
    min_eigval = ProperFractionProperty(default=1e-2)

    def init_train_updates(self):
        step = self.variables.step
        inv_min_eigval = 1 / self.min_eigval
        parameters = parameter_values(self.connection)
        param_vector = make_single_vector(parameters)

        gradients = tf.gradients(self.variables.error_func, parameters)
        full_gradient = make_single_vector(gradients)

        second_derivatives = []
        for parameter, gradient in zip(parameters, gradients):
            second_derivative, = tf.gradients(gradient, parameter)
            second_derivatives.append(flatten(second_derivative))

        hessian_diag = tf.concat(second_derivatives, axis=0)

        # it's easier to clip inverse hessian rather than the hessian,.
        inv_hessian_diag = tf.clip_by_value(
            # inverse for diagonal matrix easy to compute with
            # elementwise inverse operation.
            1 / hessian_diag,
            -inv_min_eigval,
            inv_min_eigval,
        )
        updates = setup_parameter_updates(
            parameters,
            param_vector - step * full_gradient * inv_hessian_diag
        )
        return updates
