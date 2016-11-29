import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

from neupy.core.config import ConfigurableABC
from neupy.core.properties import IntProperty, ParameterProperty
from neupy.algorithms.base import BaseNetwork
from neupy.algorithms.constructor import BaseAlgorithm
from neupy.algorithms.gd.base import (MinibatchTrainingMixin,
                                      average_batch_errors)
from neupy.layers.base import create_shared_parameter
from neupy.utils import theano_random_stream, asint, asfloat, format_data
from neupy import init


__all__ = ('RBM',)


class RBM(BaseAlgorithm, BaseNetwork, MinibatchTrainingMixin):
    """
    Boolean/Bernoulli Restricted Boltzmann Machine (RBM).
    Algorithm assumes that inputs are either binary
    values or values between 0 and 1.

    Parameters
    ----------
    n_visible : int
        Number of visible units.

    n_hidden : int
        Number of hidden units.

    {MinibatchTrainingMixin.batch_size}

    weight : array-like, Theano variable, Initializer or scalar
        Default initialization methods
        you can find :ref:`here <init-methods>`.
        Defaults to :class:`XavierNormal <neupy.init.XavierNormal>`.

    hidden_bias : array-like, Theano variable, Initializer or scalar
        Default initialization methods
        you can find :ref:`here <init-methods>`.
        Defaults to :class:`Constant(value=0) <neupy.init.Constant>`.

    visible_bias : array-like, Theano variable, Initializer or scalar
        Default initialization methods
        you can find :ref:`here <init-methods>`.
        Defaults to :class:`Constant(value=0) <neupy.init.Constant>`.

    {BaseNetwork.Parameters}

    Methods
    -------
    train(input_train, epochs=100)
        Trains network.

    {BaseSkeleton.fit}

    visible_to_hidden(visible_input)
        Populates data throught the network and returns output
        from the hidden layer.

    hidden_to_visible(hidden_input)
        Propagates output from the hidden layer backward
        to the visible.

    gibbs_sampling(visible_input, n_iter=1)
        Makes Gibbs sampling ``n`` times using visible input.

    Examples
    --------
    >>> import numpy as np
    >>> from neupy import algorithms
    >>>
    >>> data = np.array([
    ...     [1, 0, 1, 0],
    ...     [1, 0, 1, 0],
    ...     [1, 0, 0, 0],  # incomplete sample
    ...     [1, 0, 1, 0],
    ...
    ...     [0, 1, 0, 1],
    ...     [0, 0, 0, 1],  # incomplete sample
    ...     [0, 1, 0, 1],
    ...     [0, 1, 0, 1],
    ...     [0, 1, 0, 1],
    ...     [0, 1, 0, 1],
    ... ])
    >>>
    >>> rbm = algorithms.RBM(n_visible=4, n_hidden=1)
    >>> rbm.train(data, epochs=100)
    >>>
    >>> hidden_states = rbm.visible_to_hidden(data)
    >>> hidden_states.round(2)
    array([[ 0.99],
           [ 0.99],
           [ 0.95],
           [ 0.99],
           [ 0.  ],
           [ 0.01],
           [ 0.  ],
           [ 0.  ],
           [ 0.  ],
           [ 0.  ]])

    References
    ----------
    [1] G. Hinton, A Practical Guide to Training Restricted
        Boltzmann Machines, 2010.
        http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """
    n_visible = IntProperty(required=True, minval=1)
    n_hidden = IntProperty(required=True, minval=1)

    weight = ParameterProperty(default=init.XavierNormal())
    hidden_bias = ParameterProperty(default=init.Constant(value=0))
    visible_bias = ParameterProperty(default=init.Constant(value=0))

    def __init__(self, n_visible, n_hidden, **options):
        self.theano_random = theano_random_stream()

        super(ConfigurableABC, self).__init__(n_hidden=n_hidden,
                                              n_visible=n_visible, **options)

        self.weight = create_shared_parameter(
            value=self.weight,
            name='algo:rbm/matrix:weight',
            shape=(n_visible, n_hidden)
        )
        self.hidden_bias = create_shared_parameter(
            value=self.hidden_bias,
            name='algo:rbm/vector:hidden-bias',
            shape=(n_hidden,),
        )
        self.visible_bias = create_shared_parameter(
            value=self.visible_bias,
            name='algo:rbm/vector:visible-bias',
            shape=(n_visible,),
        )

        super(RBM, self).__init__(**options)

    def init_input_output_variables(self):
        self.variables.update(
            network_input=T.matrix(name='algo:rbm/var:network-input'),
        )

    def init_variables(self):
        self.variables.update(
            h_samples=theano.shared(
                name='algo:rbm/matrix:hidden-samples',
                value=asint(np.zeros((self.batch_size, self.n_hidden))),
            ),
        )

    def init_methods(self):
        def free_energy(visible_sample):
            wx_b = T.dot(visible_sample, self.weight) + self.hidden_bias
            visible_bias_term = T.dot(visible_sample, self.visible_bias)
            hidden_term = T.log(asfloat(1) + T.exp(wx_b)).sum(axis=1)
            return -visible_bias_term - hidden_term

        def visible_to_hidden(visible_sample):
            wx_b = T.dot(visible_sample, self.weight) + self.hidden_bias
            return T.nnet.sigmoid(wx_b)

        def hidden_to_visible(hidden_sample):
            wx_b = T.dot(hidden_sample, self.weight.T) + self.visible_bias
            return T.nnet.sigmoid(wx_b)

        def sample_hidden_from_visible(visible_sample):
            theano_random = self.theano_random
            hidden_prob = visible_to_hidden(visible_sample)
            hidden_sample = theano_random.binomial(n=1, p=hidden_prob,
                                                   dtype=theano.config.floatX)
            return hidden_sample

        def sample_visible_from_hidden(hidden_sample):
            theano_random = self.theano_random
            visible_prob = hidden_to_visible(hidden_sample)
            visible_sample = theano_random.binomial(n=1, p=visible_prob,
                                                    dtype=theano.config.floatX)
            return visible_sample

        network_input = self.variables.network_input
        n_samples = asfloat(network_input.shape[0])
        theano_random = self.theano_random

        weight = self.weight
        h_bias = self.hidden_bias
        v_bias = self.visible_bias
        h_samples = self.variables.h_samples
        step = asfloat(self.step)

        sample_indeces = theano_random.random_integers(
            low=0, high=n_samples - 1,
            size=(self.batch_size,)
        )
        v_pos = ifelse(
            T.eq(n_samples, self.batch_size),
            network_input,
            # In case if final batch has less number of
            # samples then expected
            network_input[sample_indeces]
        )
        h_pos = visible_to_hidden(v_pos)

        v_neg = sample_visible_from_hidden(h_samples)
        h_neg = visible_to_hidden(v_neg)

        weight_update = v_pos.T.dot(h_pos) - v_neg.T.dot(h_neg)
        h_bias_update = (h_pos - h_neg).mean(axis=0)
        v_bias_update = (v_pos - v_neg).mean(axis=0)

        # Stochastic pseudo-likelihood
        feature_index_to_flip = theano_random.random_integers(
            low=0,
            high=self.n_visible - 1,
        )
        rounded_input = T.round(network_input)
        rounded_input = network_input
        rounded_input_flip = T.set_subtensor(
            rounded_input[:, feature_index_to_flip],
            1 - rounded_input[:, feature_index_to_flip]
        )
        error = T.mean(
            self.n_visible * T.log(T.nnet.sigmoid(
                free_energy(rounded_input_flip) -
                free_energy(rounded_input)
            ))
        )

        self.methods.update(
            train_epoch=theano.function(
                [network_input],
                error,
                name='algo:rbm/func:train-epoch',
                updates=[
                    (weight, weight + step * weight_update / n_samples),
                    (h_bias, h_bias + step * h_bias_update),
                    (v_bias, v_bias + step * v_bias_update),
                    (h_samples, asint(theano_random.binomial(n=1, p=h_neg))),
                ]
            ),
            prediction_error=theano.function(
                [network_input], error,
                name='algo:rbm/func:prediction-error',
            ),
            visible_to_hidden=theano.function(
                [network_input],
                visible_to_hidden(network_input),
                name='algo:rbm/func:visible-to-hidden',
            ),
            hidden_to_visible=theano.function(
                [network_input],
                hidden_to_visible(network_input),
                name='algo:rbm/func:hidden-to-visible',
            ),
            gibbs_sampling=theano.function(
                [network_input],
                sample_visible_from_hidden(
                    sample_hidden_from_visible(network_input)
                ),
                name='algo:rbm/func:gibbs-sampling',
            )
        )

    def train(self, input_train, input_test=None, epochs=100,
              summary='table'):
        """
        Train RBM.

        Parameters
        ----------
        input_train : 1D or 2D array-like
        input_test : 1D or 2D array-like or None
            Defaults to ``None``.
        epochs : int
            Number of training epochs. Defaults to ``100``.
        summary : {'table', 'inline'}
            Training summary type. Defaults to ``'table'``.
        """
        return super(RBM, self).train(
            input_train=input_train, target_train=None,
            input_test=input_test, target_test=None,
            epochs=epochs, epsilon=None, summary=summary
        )

    def train_epoch(self, input_train, target_train=None):
        """
        Train one epoch.

        Parameters
        ----------
        input_train : array-like (n_samples, n_features)

        Returns
        -------
        float
        """
        errors = self.apply_batches(
            function=self.methods.train_epoch,
            input_data=input_train,

            description='Training batches',
            show_error_output=True,
        )

        n_samples = len(input_train)
        return average_batch_errors(errors, n_samples, self.batch_size)

    def visible_to_hidden(self, visible_input):
        """
        Populates data throught the network and returns output
        from the hidden layer.

        Parameters
        ----------
        visible_input : array-like (n_samples, n_visible_features)

        Returns
        -------
        array-like
        """
        is_input_feature1d = (self.n_visible == 1)
        visible_input = format_data(visible_input, is_input_feature1d)

        outputs = self.apply_batches(
            function=self.methods.visible_to_hidden,
            input_data=visible_input,

            description='Hidden from visible batches',
            show_progressbar=True,
            show_error_output=False,
        )

        return np.concatenate(outputs, axis=0)

    def hidden_to_visible(self, hidden_input):
        """
        Propagates output from the hidden layer backward
        to the visible.

        Parameters
        ----------
        hidden_input : array-like (n_samples, n_hidden_features)

        Returns
        -------
        array-like
        """
        is_input_feature1d = (self.n_hidden == 1)
        hidden_input = format_data(hidden_input, is_input_feature1d)

        outputs = self.apply_batches(
            function=self.methods.hidden_to_visible,
            input_data=hidden_input,

            description='Visible from hidden batches',
            show_progressbar=True,
            show_error_output=False,
        )

        return np.concatenate(outputs, axis=0)

    def prediction_error(self, input_data, target_data=None):
        """
        Compute the pseudo-likelihood of input samples.

        Parameters
        ----------
        input_data : array-like
            Values of the visible layer

        Returns
        -------
        float
            Value of the pseudo-likelihood.
        """
        is_input_feature1d = (self.n_visible == 1)
        input_data = format_data(input_data, is_input_feature1d)

        errors = self.apply_batches(
            function=self.methods.prediction_error,
            input_data=input_data,

            description='Validation batches',
            show_error_output=True,
        )
        return average_batch_errors(
            errors,
            n_samples=len(input_data),
            batch_size=self.batch_size,
        )

    def gibbs_sampling(self, visible_input, n_iter=1):
        """
        Makes Gibbs sampling n times using visible input.

        Parameters
        ----------
        visible_input : 1d or 2d array
        n_iter : int
            Number of Gibbs sampling iterations. Defaults to ``1``.

        Returns
        -------
        array-like
            Output from the visible units after perfoming n
            Gibbs samples. Array will contain only binary
            units (0 and 1).
        """
        is_input_feature1d = (self.n_visible == 1)
        visible_input = format_data(visible_input, is_input_feature1d)

        gibbs_sampling = self.methods.gibbs_sampling

        input_ = visible_input
        for iteration in range(n_iter):
            input_ = gibbs_sampling(input_)

        return input_
