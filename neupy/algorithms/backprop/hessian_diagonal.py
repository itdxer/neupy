from __future__ import division

from numpy import dot, asmatrix, reshape, where
from scipy.sparse import lil_matrix

from neupy.core.properties import BetweenZeroAndOneProperty
from .backpropagation import Backpropagation


__all__ = ('HessianDiagonal',)


class HessianDiagonal(Backpropagation):
    """ Hissian diagonal is a Hessian algorithm approximation which require
    only computation of hessian matrix diagonal elements and makes it
    invertion much easier and faster.

    Parameters
    ----------
    min_eigenvalue : float
        Setup min eigenvalue for Hessian diagonale matrix. After few
        iteration elements would be extremly small and matrix inverse
        produce huge number in hessian diagonal elements. This
        parameter control diagonal elements size. Defaults to ``1e-10``.
    {optimizations}
    {raw_predict_param}
    {full_params}

    Methods
    -------
    {supervised_train}
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
    ...         layers.SigmoidLayer(10),
    ...         layers.SigmoidLayer(20),
    ...         layers.OutputLayer(1)
    ...     ],
    ...     step=1.5,
    ...     use_raw_predict_at_error=False,
    ...     shuffle_data=False,
    ...     verbose=False,
    ...     min_eigenvalue=1e-10
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
    :network:`Backpropagation` : Backpropagation algorithm.
    """
    min_eigenvalue = BetweenZeroAndOneProperty(default=1e-10)

    def get_weight_delta(self, output_train, target_train):
        weight_deltas = []
        gradients = self.gradients = []
        state_delta = self.delta = []

        update_first_order = self.error.deriv(output_train, target_train)
        min_eigenvalue = self.min_eigenvalue
        prev_weight = None
        update_second_order = None

        for i, layer in enumerate(reversed(self.train_layers), start=1):
            summated_data = self.summated_data[-i]
            current_layer_input = self.layer_outputs[-i]
            weight = layer.weight_without_bias.T
            weight_shape = layer.weight.shape

            activation_function_deriv = layer.activation_function.deriv
            deriv = activation_function_deriv(summated_data)
            second_deriv = activation_function_deriv.deriv(summated_data)

            if i == 1:
                # For last layer update
                delta = deriv ** 2 - update_first_order * second_deriv
            else:
                # For the hidden layers
                update_first_order = update_first_order.dot(prev_weight)
                delta = (
                    deriv ** 2 * update_second_order +
                    update_first_order * second_deriv
                )

            update_first_order *= deriv
            update_second_order = delta.dot(weight ** 2)

            # Compute gradient
            gradient = current_layer_input.T.dot(update_first_order).ravel()
            gradients.insert(0, reshape(gradient, weight_shape))

            # Compute hessian matrix
            weight_delta = asmatrix(dot(current_layer_input.T ** 2, delta))
            hessain_shape = (weight_delta.size, weight_delta.size)
            inverted_hessian = lil_matrix(hessain_shape)
            # Inverse for diagonal matrix is just reciprocal
            # every diagonal element
            full_gradients = weight_delta.ravel().T
            full_gradients = where(
                full_gradients < min_eigenvalue,
                min_eigenvalue,
                full_gradients
            )
            inverted_hessian.setdiag(1 / weight_delta.ravel().T)
            weight_delta = inverted_hessian.dot(gradient)

            weight_deltas.insert(0, reshape(-weight_delta, weight_shape))
            state_delta.insert(0, delta)
            prev_weight = weight

        return weight_deltas
