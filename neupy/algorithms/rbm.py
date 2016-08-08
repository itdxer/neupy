import theano
import theano.tensor as T
import numpy as np

from neupy.core.properties import IntProperty, ParameterProperty
from neupy.network.constructor import BaseAlgorithm
from neupy.network.base import BaseNetwork
from neupy.network.learning import UnsupervisedLearningMixin
from neupy.algorithms.gd.base import (BatchSizeProperty, average_batch_errors,
                                      cannot_divide_into_batches,
                                      apply_batches)
from neupy.layers.base import create_shared_parameter
from neupy.utils import theano_random_stream, asint, asfloat, format_data
from neupy import init


__all__ = ('RBM',)


class RBM(BaseAlgorithm, BaseNetwork, UnsupervisedLearningMixin):
    """
    Boolean/Bernoulli Restricted Boltzmann Machine (RBM).

    Parameters
    ----------
    n_visible : int
        Number of visible units.
    n_hidden : int
        Number of hidden units.
    batch_size : int or {{None, -1, 'all', '*', 'full'}}
        Set up batch size for learning process. To set up batch size equal to
        sample size value should be equal to one of the values listed above.
        Defaults to ``128``.
    n_gibbs_steps : int
        Number of Gibbs samples that algorithm need to perfom during
        each epoch. Defaults to ``1``.
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
    {UnsupervisedLearningMixin.Methods}

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

    n_gibbs_steps = IntProperty(default=1, minval=1)
    batch_size = BatchSizeProperty(default=10)

    def __init__(self, n_hidden, **options):
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

    def init_methods(self):
        network_input = self.variables.network_input
        n_samples = asfloat(network_input.shape[0])
        theano_random = theano_random_stream()

        weight = self.weight
        h_bias = self.hidden_bias
        v_bias = self.visible_bias
        h_samples = self.variables.h_samples
        step = asfloat(self.step)

        v_pos = network_input
        h_pos = self.hidden_from_visible(v_pos)

        v_neg_prob = self.visible_from_hidden(h_samples)
        v_neg = theano_random.binomial(n=1, p=v_neg_prob,
                                       dtype=theano.config.floatX)
        h_neg = self.hidden_from_visible(v_neg)

        weight_update = v_pos.T.dot(h_pos) - v_neg.T.dot(h_neg)
        h_bias_update = (h_pos - h_neg).mean(axis=0)
        v_bias_update = (v_pos - v_neg).mean(axis=0)

        error = T.mean(self.free_energy(v_pos) - self.free_energy(v_neg))

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
            hidden_from_visible=theano.function(
                [network_input],
                self.hidden_from_visible(network_input)
            ),
        )

    def train_epoch(self, input_train, target_train):
        train_epoch = self.methods.train_epoch

        if cannot_divide_into_batches(input_train, self.batch_size):
            return train_epoch(input_train)

        show_progressbar = (self.training and self.training.show_epoch == 1)
        errors = apply_batches(
            function=train_epoch,
            arguments=(input_train,),
            batch_size=self.batch_size,

            description='Training batches',
            logger=self.logs,
            show_progressbar=show_progressbar,
            show_error_output=True,
        )
        return average_batch_errors(
            errors,
            n_samples=len(input_train),
            batch_size=self.batch_size,
        )

    def transform(self, input_data):
        """
        Populates data throught the network and returns output
        from the hidden layer.

        Parameters
        ----------
        input_data : array-like

        Returns
        -------
        array-like
        """
        is_input_feature1d = (self.n_visible == 1)
        input_data = format_data(input_data, is_input_feature1d)
        hidden_from_visible = self.methods.hidden_from_visible

        if cannot_divide_into_batches(input_data, self.batch_size):
            return hidden_from_visible(input_data)

        outputs = apply_batches(
            function=hidden_from_visible,
            arguments=(input_data,),
            batch_size=self.batch_size,

            description='Transformation batches',
            logger=self.logs,
            show_progressbar=True,
            show_error_output=False,
        )

        return np.concatenate(outputs, axis=0)
