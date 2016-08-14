import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

from neupy.core.properties import IntProperty, ParameterProperty
from neupy.network.constructor import BaseAlgorithm
from neupy.network.base import BaseNetwork
from neupy.network.learning import UnsupervisedLearningMixin
from neupy.algorithms.gd.base import (MinibatchTrainingMixin,
                                      average_batch_errors)
from neupy.layers.base import create_shared_parameter
from neupy.utils import theano_random_stream, asint, asfloat, format_data
from neupy import init


__all__ = ('RBM',)


class RBM(UnsupervisedLearningMixin, BaseAlgorithm, BaseNetwork,
          MinibatchTrainingMixin):
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
    n_gibbs_steps : int
        Number of Gibbs samples that algorithm need to perfom during
        the training procedure. Defaults to ``1``.
    weight : array-like, Theano variable, Initializer or scalar
        Default initialization methods
        you can find :ref:`here <init-methods>`.
        Defaults to :class:`XavierNormal <neupy.core.init.XavierNormal>`.
    hidden_bias : array-like, Theano variable, Initializer or scalar
        Default initialization methods
        you can find :ref:`here <init-methods>`.
        Defaults to :class:`Constant(value=0) <neupy.core.init.Constant>`.
    visible_bias : array-like, Theano variable, Initializer or scalar
        Default initialization methods
        you can find :ref:`here <init-methods>`.
        Defaults to :class:`Constant(value=0) <neupy.core.init.Constant>`.
    {BaseNetwork.Parameters}

    Methods
    -------
    {UnsupervisedLearningMixin.train}
    {BaseSkeleton.fit}
    transform(input_data)
        Propagates input data through the network and return
        output from the hidden units.
    predict(input_data)
        Alias to ``transform`` method.
    gibbs_sampling(visible_input, n_iter=1)
        Makes Gibbs sampling n times using visible input.

    References
    ----------
    [1] G. Hinton, A Practical Guide to Training Restricted
        Boltzmann Machines, 2010.
        http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """
    n_visible = IntProperty(required=True, minval=1)
    n_hidden = IntProperty(required=True, minval=1)

    n_gibbs_steps = IntProperty(default=1, minval=1)

    weight = ParameterProperty(default=init.XavierNormal())
    hidden_bias = ParameterProperty(default=init.Constant(value=0))
    visible_bias = ParameterProperty(default=init.Constant(value=0))

    def __init__(self, n_hidden, **options):
        self.theano_random = theano_random_stream()
        super(RBM, self).__init__(n_hidden=n_hidden, **options)

    def init_layers(self):
        n_hidden = self.n_hidden
        n_visible = self.n_visible

        self.weight = create_shared_parameter(
            value=self.weight,
            name='weight',
            shape=(n_visible, n_hidden)
        )
        self.hidden_bias = create_shared_parameter(
            value=self.hidden_bias,
            name='hidden_bias',
            shape=(n_hidden,),
        )
        self.visible_bias = create_shared_parameter(
            value=self.visible_bias,
            name='visible_bias',
            shape=(n_visible,),
        )

    def init_input_output_variables(self):
        self.variables.update(
            network_input=T.matrix(name='network_input'),
        )

    def init_variables(self):
        self.init_layers()

        self.variables.update(
            h_samples=theano.shared(
                name='h_samples',
                value=asint(np.zeros((self.batch_size, self.n_hidden))),
            ),
        )

    def free_energy(self, visible_sample):
        wx_b = T.dot(visible_sample, self.weight) + self.hidden_bias
        visible_bias_term = T.dot(visible_sample, self.visible_bias)
        hidden_term = T.log(asfloat(1) + T.exp(wx_b)).sum(axis=1)
        return -visible_bias_term - hidden_term

    def hidden_from_visible(self, visible_sample):
        wx_b = T.dot(visible_sample, self.weight) + self.hidden_bias
        return T.nnet.sigmoid(wx_b)

    def visible_from_hidden(self, hidden_sample):
        wx_b = T.dot(hidden_sample, self.weight.T) + self.visible_bias
        return T.nnet.sigmoid(wx_b)

    def sample_hidden_from_visible(self, visible_sample):
        theano_random = self.theano_random
        hidden_prob = self.hidden_from_visible(visible_sample)
        hidden_sample = theano_random.binomial(n=1, p=hidden_prob,
                                               dtype=theano.config.floatX)
        return hidden_sample

    def sample_visible_from_hidden(self, hidden_sample):
        theano_random = self.theano_random
        visible_prob = self.visible_from_hidden(hidden_sample)
        visible_sample = theano_random.binomial(n=1, p=visible_prob,
                                                dtype=theano.config.floatX)
        return visible_sample

    def init_methods(self):
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
        h_pos = self.hidden_from_visible(v_pos)

        v_neg = self.sample_visible_from_hidden(h_samples)
        h_neg = self.hidden_from_visible(v_neg)

        weight_update = v_pos.T.dot(h_pos) - v_neg.T.dot(h_neg)
        h_bias_update = (h_pos - h_neg).mean(axis=0)
        v_bias_update = (v_pos - v_neg).mean(axis=0)

        # Stochastic pseudo-likelihood
        feature_index_to_flip = theano_random.random_integers(
            low=0,
            high=self.n_visible - 1,
        )
        # rounded_input = T.round(network_input)
        rounded_input = network_input
        rounded_input_flip = T.set_subtensor(
            rounded_input[:, feature_index_to_flip],
            1 - rounded_input[:, feature_index_to_flip]
        )
        error = T.mean(
            self.n_visible * T.log(T.nnet.sigmoid(
                self.free_energy(rounded_input_flip) -
                self.free_energy(rounded_input)
            ))
        )

        self.methods.update(
            train_epoch=theano.function(
                [network_input],
                error,
                updates=[
                    (weight, weight + step * weight_update / n_samples),
                    (h_bias, h_bias + step * h_bias_update),
                    (v_bias, v_bias + step * v_bias_update),
                    (h_samples, asint(theano_random.binomial(n=1, p=h_neg))),
                ]
            ),
            prediction_error=theano.function([network_input], error),
            hidden_from_visible=theano.function(
                [network_input],
                self.hidden_from_visible(network_input)
            ),
            gibbs_sampling=theano.function(
                [network_input],
                self.visible_from_hidden(
                    self.sample_hidden_from_visible(network_input)
                )
            )
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

    def transform(self, input_data):
        """
        Populates data throught the network and returns output
        from the hidden layer.

        Parameters
        ----------
        input_data : array-like (n_samples, n_features)

        Returns
        -------
        array-like
        """
        is_input_feature1d = (self.n_visible == 1)
        input_data = format_data(input_data, is_input_feature1d)

        outputs = self.apply_batches(
            function=self.methods.hidden_from_visible,
            input_data=input_data,

            description='Transformation batches',
            show_progressbar=True,
            show_error_output=False,
        )

        return np.concatenate(outputs, axis=0)

    def predict(self, input_data):
        """
        Alias to ``transform`` method.
        """
        return self.transform(input_data)

    def prediction_error(self, input_data, target_data=None):
        """
        Check the prediction error for the specified input samples
        and their targets.

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        float
            Prediction error.
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
        visible_input : array-like
        n_iter : int
            Number of Gibbs sampling iterations. Defaults to ``1``.

        Returns
        -------
        array-like
            Output from the visible units after perfoming n
            Gibbs samples.
        """
        is_input_feature1d = (self.n_visible == 1)
        visible_input = format_data(visible_input, is_input_feature1d)

        gibbs_sampling = self.methods.gibbs_sampling

        input_ = visible_input
        for iteration in range(n_iter):
            input_ = gibbs_sampling(input_)

        return input_
