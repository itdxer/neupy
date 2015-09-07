from operator import mul

from numpy import zeros, asmatrix, identity, tile, dot, concatenate

from neupy.core.properties import NonNegativeNumberProperty
from neupy.functions import mse
from neupy.algorithms import Backpropagation


__all__ = ('LevenbergMarquardt',)


class LevenbergMarquardt(Backpropagation):
    """ Levenberg-Marquardt algorithm.

    Parameters
    ----------
    mu : float
        Control invertion for J.T * J matrix, defaults to `0.1`.
    mu_increase_factor : float
        Factor to decrease the mu if update decrese the error, otherwise
        increse mu by the same factor.
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
    >>> lmnet = algorithms.LevenbergMarquardt(
    ...     (2, 3, 1),
    ...     verbose=False
    ... )
    >>> lmnet.train(x_train, y_train)

    Diabets dataset example

    >>> import numpy as np
    >>> from sklearn import datasets, preprocessing
    >>> from sklearn.cross_validation import train_test_split
    >>> from neupy import algorithms, layers
    >>> from neupy.functions import rmsle
    >>>
    >>> dataset = datasets.load_diabetes()
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
    >>> # Network
    ... lmnet = algorithms.LevenbergMarquardt(
    ...     connection=[
    ...         layers.SigmoidLayer(10),
    ...         layers.SigmoidLayer(40),
    ...         layers.OutputLayer(1),
    ...     ],
    ...     mu_increase_factor=2,
    ...     mu=0.1,
    ...     step=0.25,
    ...     show_epoch=10,
    ...     use_bias=False,
    ...     verbose=False
    ... )
    >>> lmnet.train(x_train, y_train, epochs=100)
    >>> y_predict = lmnet.predict(x_test)
    >>>
    >>> error = rmsle(target_scaler.inverse_transform(y_test),
    ...               target_scaler.inverse_transform(y_predict).round())
    >>> error
    0.47548200957888398

    See Also
    --------
    :network:`Backpropagation` : Backpropagation algorithm.
    """
    mu = NonNegativeNumberProperty(default=0.01)
    mu_increase_factor = NonNegativeNumberProperty(default=5, min_size=1)

    def setup_defaults(self):
        super(LevenbergMarquardt, self).setup_defaults()
        # Can use only squared error
        del self.error
        self.error = mse  # `error` isn't a property instance after deletion.

    def init_layers(self):
        super(LevenbergMarquardt, self).init_layers()
        self.n_weights = sum(mul(*layer.size) for layer in self.train_layers)

    def get_jacobian_matrix(self):
        result = []
        for input_data, gradient in zip(self.layer_outputs, self.delta):
            gradient_n_col = gradient.shape[1]
            input_data_n_features = input_data.shape[1]

            repeated_gradients = tile(gradient, (1, input_data_n_features))
            # Total weight matrix size (rows*cols) in layer.
            n_weights = repeated_gradients.shape[1]

            # Matrix `comb` help repeate input data columns for valid
            # combination of input data and gradients for Jacobian matrix.
            # Basicly this matrix and dot product replace 2 inner loops
            # in pure python.
            comb = zeros((input_data_n_features, n_weights))
            for i in range(input_data_n_features):
                comb[i, i * gradient_n_col:(i + 1) * gradient_n_col] = 1

            result.append(repeated_gradients * dot(input_data, comb))

        return concatenate(result, axis=1)

    def update_weights(self, weight_deltas):
        error = asmatrix(self.target_train - self.output_train)
        jacoby = self.get_jacobian_matrix() / error
        jacoby_t = jacoby.T
        smallstep_matrix = self.mu * identity(self.n_weights)

        inverse_hessian = (jacoby_t * jacoby + smallstep_matrix).I
        # Most of all time this order would be much faster than sequantial
        # order without brackets, because number of columns for parameter
        # error would be small.
        jacoby_delta = inverse_hessian * (jacoby_t * error)

        row = 0
        old_weights = self.old_weights = []

        for layer in self.train_layers:
            weight = layer.weight
            weight_in_size, weight_out_size = weight.shape
            old_weights.append(weight.copy())

            for i in range(self.use_bias, weight_in_size):
                weight_delta = jacoby_delta[row:row + weight_out_size, :].T
                weight[i:i + 1, :] -= weight_delta
                row += weight_out_size

            if self.use_bias:
                # Use otside if loop because we setup last column for bias
                # which is at fist row position in weight matrix.
                weight[0:1, :] -= jacoby_delta[row:row + weight_out_size, :].T
                row += weight_out_size

    def after_weight_update(self, input_train, target_train):
        super(LevenbergMarquardt, self).after_weight_update(
            input_train, target_train
        )

        if not self.last_error():
            return

        output_train = self.predict(input_train)
        error = self.error(output_train, target_train)

        if error < self.last_error_in():
            self.mu /= self.mu_increase_factor
            return

        self.mu *= self.mu_increase_factor

        # Error changed in wrong way. Rollback weight updates.
        for i, layer in enumerate(self.train_layers):
            layer.weight = self.old_weights[i]
