import numpy as np
import tensorflow as tf

from neupy import init
from neupy.utils import AttributeKeyDict, as_tuple, tensorflow_session
from neupy.exceptions import LayerConnectionError
from neupy.core.properties import (IntProperty, Property, NumberProperty,
                                   ParameterProperty)
from .base import BaseLayer


__all__ = ('LSTM', 'GRU')


def clip_gradient(value, clip_value):
    if not hasattr(clip_gradient, 'added_gradients'):
        clip_gradient.added_gradients = set()

    session = tensorflow_session()
    graph = session.graph
    operation_name = "ClipGradient-" + str(clip_value)

    if operation_name not in clip_gradient.added_gradients:
        # Make sure that we won't create the same operation twise.
        # Otherwise tensorflow will trigger an exception.
        @tf.RegisterGradient(operation_name)
        def clip_gradient_grad(op, grad):
            return tf.clip_by_value(grad, -clip_value, clip_value)

        clip_gradient.added_gradients.add(operation_name)

    with graph.gradient_override_map({"Identity": operation_name}):
        return tf.identity(value)


def unroll_scan(fn, sequence, outputs_info):
    """
    Helper function to unroll for loops. Can be used to unroll
    ``tensorflow.scan``.

    Parameters
    ----------
    fn : function
        Function that defines calculations at each step.

    sequences : TensorVariable or list of TensorVariables
        List of TensorVariable with sequence data. The function iterates
        over the first dimension of each TensorVariable.

    outputs_info : list of TensorVariables
        List of tensors specifying the initial values for each recurrent
        value.

    Returns
    -------
    List of TensorVariables. Each element in the list gives
    the recurrent values at each time step.
    """
    with tf.name_scope('unroll-scan'):
        outputs = []
        prev_vals = outputs_info

        for entity in tf.unstack(sequence):
            output = fn(prev_vals, entity)
            outputs.append(output[-1])
            prev_vals = output

        return tf.stack(outputs)


class MultiCallableProperty(ParameterProperty):
    expected_type = as_tuple(dict)

    def __set__(self, instance, value):
        self.validate(value)

        default_value = self.default.copy()
        default_value.update(value)
        value = default_value

        value = AttributeKeyDict(value)
        instance.__dict__[self.name] = value

    def validate(self, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                "Parameter `{}` should be a dictionary, "
                "got `{!r}`".format(self.name, type(value)))

        for key, func in value.items():
            if not callable(func):
                raise ValueError(
                    "Values for the `{}` parameter should be "
                    "callable objects, got value `{!r}` for the "
                    "`{}` key".format(self.name, func, key))


class BaseRNNLayer(BaseLayer):
    """
    Base class for the recurrent layers

    Parameters
    ----------
    size : int
        Number of hidden units in the layer.

    only_return_final : bool
        If ``True``, only return the final sequential output
        (e.g. for tasks where a single target value for the entire
        sequence is desired). In this case, Tensorfow makes an
        optimization which saves memory. Defaults to ``True``.

    {BaseLayer.Parameters}
    """
    size = IntProperty(minval=1)
    only_return_final = Property(default=True, expected_type=bool)

    def __init__(self, size, **kwargs):
        super(BaseRNNLayer, self).__init__(size=size, **kwargs)

    def validate(self, input_shape):
        n_input_dims = len(input_shape) + 1  # +1 for batch dimension
        clsname = self.__class__.__name__

        if n_input_dims != 3:
            raise LayerConnectionError(
                "{} layer was expected input with three dimensions, "
                "but got input with {} dimensions instead"
                "".format(clsname, n_input_dims))

    @property
    def output_shape(self):
        if self.only_return_final:
            return as_tuple(self.size)

        n_time_steps = self.input_shape[0]
        return as_tuple(n_time_steps, self.size)


class LSTM(BaseRNNLayer):
    """
    Long Short Term Memory (LSTM) Layer.

    Parameters
    ----------
    {BaseRNNLayer.size}

    input_weights : Initializer, ndarray
        Weight parameters for input connection.
        Defaults to :class:`HeNormal() <neupy.init.HeNormal>`.

    hidden_weights : Initializer, ndarray
        Weight parameters for hidden connection.
        Defaults to :class:`HeNormal() <neupy.init.HeNormal>`.

    cell_weights : Initializer, ndarray
        Weight parameters for cell connection. Require only when
        ``peepholes=True`` otherwise it will be ignored.
        Defaults to :class:`HeNormal() <neupy.init.HeNormal>`.

    bias : Initializer, ndarray
        Bias parameters for all gates.
        Defaults to :class:`Constant(0) <neupy.init.Constant>`.

    activation_functions : dict, callable
        Activation functions for different gates. Defaults to:

        .. code-block:: python

            # import tensorflow as tf
            dict(
                ingate=tf.nn.sigmoid,
                forgetgate=tf.nn.sigmoid,
                outgate=tf.nn.sigmoid,
                cell=tf.tanh,
            )

        If application requires modification to only one parameter
        then it's better to specify the one that you need to modify
        and ignore other parameters

        .. code-block:: python

            dict(ingate=tf.tanh)

        Other parameters like ``forgetgate`` or ``outgate`` will be
        equal to their default values.

    learn_init : bool
        If ``True``, make ``cell_init`` and ``hidden_init`` trainable
        variables. Defaults to ``False``.

    cell_init : array-like, Tensorfow variable, scalar or Initializer
        Initializer for initial cell state (:math:`c_0`).
        Defaults to :class:`Constant(0) <neupy.init.Constant>`.

    hidden_init : array-like, Tensorfow variable, scalar or Initializer
        Initializer for initial hidden state (:math:`h_0`).
        Defaults to :class:`Constant(0) <neupy.init.Constant>`.

    backwards : bool
        If ``True``, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`. Defaults to ``False``

    {BaseRNNLayer.only_return_final}

    peepholes : bool
        If ``True``, the LSTM uses peephole connections.
        When ``False``, cell parameters  are ignored.
        Defaults to ``False``.

    unroll_scan : bool
        If ``True`` the recursion is unrolled instead of using scan.
        For some graphs this gives a significant speed up but it
        might also consume more memory. When ``unroll_scan=True``,
        backpropagation always includes the full sequence, so
        ``n_gradient_steps`` must be set to ``-1`` and the input
        sequence length must be known at compile time (i.e.,
        cannot be given as ``None``). Defaults to ``False``.

    gradient_clipping : float or int
        If nonzero, the gradient messages are clipped to the
        given value during the backward pass. Defaults to ``0``.

    {BaseLayer.Parameters}

    Notes
    -----
    Code was adapted from the
    `Lasagne <https://github.com/Lasagne/Lasagne>`_ library.

    Examples
    --------

    Sequence classification

    .. code-block:: python

        from neupy import layers, algorithms

        n_time_steps = 40
        n_categories = 20
        embedded_size = 10

        network = algorithms.RMSProp(
            [
                layers.Input(n_time_steps),
                layers.Embedding(n_categories, embedded_size),
                layers.LSTM(20),
                layers.Sigmoid(1),
            ]
        )
    """
    input_weights = ParameterProperty(default=init.HeNormal())
    hidden_weights = ParameterProperty(default=init.HeNormal())
    cell_weights = ParameterProperty(default=init.HeNormal())
    biases = ParameterProperty(default=init.Constant(0))

    activation_functions = MultiCallableProperty(
        default=dict(
            ingate=tf.nn.sigmoid,
            forgetgate=tf.nn.sigmoid,
            outgate=tf.nn.sigmoid,
            cell=tf.tanh,
        )
    )

    learn_init = Property(default=False, expected_type=bool)
    cell_init = ParameterProperty(default=init.Constant(0))
    hidden_init = ParameterProperty(default=init.Constant(0))

    unroll_scan = Property(default=False, expected_type=bool)
    backwards = Property(default=False, expected_type=bool)
    peepholes = Property(default=False, expected_type=bool)
    gradient_clipping = NumberProperty(default=0, minval=0)

    def initialize(self):
        super(LSTM, self).initialize()
        n_inputs = np.prod(self.input_shape[1:])

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.weight_cell_to_ingate = self.add_parameter(
                value=self.cell_weights,
                name='weight_cell_to_ingate',
                shape=(self.size,))
            self.weight_cell_to_forgetgate = self.add_parameter(
                value=self.cell_weights,
                name='weight_cell_to_forgetgate',
                shape=(self.size,))
            self.weight_cell_to_outgate = self.add_parameter(
                value=self.cell_weights,
                name='weight_cell_to_outgate',
                shape=(self.size,))

        self.input_weights = self.add_parameter(
            value=self.input_weights,
            name='input_weights',
            shape=(n_inputs, 4 * self.size),
        )
        self.hidden_weights = self.add_parameter(
            value=self.hidden_weights,
            name='hidden_weights',
            shape=(self.size, 4 * self.size),
        )
        self.biases = self.add_parameter(
            value=self.biases, name='biases',
            shape=(4 * self.size,),
        )

        # Initialization parameters
        self.add_parameter(
            value=self.cell_init,
            shape=(1, self.size),
            name="cell_init",
            trainable=self.learn_init,
        )
        self.add_parameter(
            value=self.hidden_init,
            shape=(1, self.size),
            name="hidden_init",
            trainable=self.learn_init,
        )

    def output(self, input_value):
        # Because scan iterates over the first dimension we
        # dimshuffle to (n_time_steps, n_batch, n_features)
        input_value = tf.transpose(input_value, [1, 0, 2])
        input_shape = tf.shape(input_value)
        n_batch = input_shape[1]

        def one_lstm_step(states, input_n):
            with tf.name_scope('lstm-cell'):
                cell_previous, hid_previous = states
                input_n = tf.matmul(input_n, self.input_weights) + self.biases

                # Calculate gates pre-activations and slice
                gates = input_n + tf.matmul(hid_previous, self.hidden_weights)

                # Clip gradients
                if self.gradient_clipping != 0:
                    gates = clip_gradient(gates, self.gradient_clipping)

                # Extract the pre-activation gate values
                ingate, forgetgate, cell_input, outgate = tf.split(
                    gates, 4, axis=1)

                if self.peepholes:
                    # Compute peephole connections
                    ingate += cell_previous * self.weight_cell_to_ingate
                    forgetgate += (
                        cell_previous * self.weight_cell_to_forgetgate)

                # Apply nonlinearities
                ingate = self.activation_functions.ingate(ingate)
                forgetgate = self.activation_functions.forgetgate(forgetgate)
                cell_input = self.activation_functions.cell(cell_input)

                # Compute new cell value
                cell = forgetgate * cell_previous + ingate * cell_input

                if self.peepholes:
                    outgate += cell * self.weight_cell_to_outgate

                outgate = self.activation_functions.outgate(outgate)

                # Compute new hidden unit activation
                hid = outgate * tf.tanh(cell)
                return [cell, hid]

        cell_init = tf.tile(self.cell_init, (n_batch, 1))
        hidden_init = tf.tile(self.hidden_init, (n_batch, 1))
        sequence = input_value

        if self.backwards:
            sequence = tf.reverse(sequence, axis=[0])

        if self.unroll_scan:
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=one_lstm_step,
                sequence=sequence,
                outputs_info=[cell_init, hidden_init],
            )
        else:
            _, hid_out = tf.scan(
                fn=one_lstm_step,
                elems=input_value,
                initializer=[cell_init, hidden_init],
                name='lstm-scan',
            )

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            return hid_out[-1]

        # if scan is backward reverse the output
        if self.backwards:
            hid_out = tf.reverse(hid_out, axis=[0])

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        hid_out = tf.transpose(hid_out, [1, 0, 2])

        return hid_out


class GRU(BaseRNNLayer):
    """
    Gated Recurrent Unit (GRU) Layer.

    Parameters
    ----------
    {BaseRNNLayer.size}

    input_weights : Initializer, ndarray
        Weight parameters for input connection.
        Defaults to :class:`HeNormal() <neupy.init.HeNormal>`.

    hidden_weights : Initializer, ndarray
        Weight parameters for hidden connection.
        Defaults to :class:`HeNormal() <neupy.init.HeNormal>`.

    bias : Initializer, ndarray
        Bias parameters for all gates.
        Defaults to :class:`Constant(0) <neupy.init.Constant>`.

    activation_functions : dict, callable
        Activation functions for different gates. Defaults to:

        .. code-block:: python

            # import tensorflow as tf
            dict(
                resetgate=tf.nn.sigmoid,
                updategate=tf.nn.sigmoid,
                hidden_update=tf.tanh,
            )

        If application requires modification to only one parameter
        then it's better to specify the one that you need to modify
        and ignore other parameters

        .. code-block:: python

            dict(resetgate=tf.tanh)

        Other parameters like ``updategate`` or ``hidden_update``
        will be equal to their default values.

    learn_init : bool
        If ``True``, make ``hidden_init`` trainable variable.
        Defaults to ``False``.

    hidden_init : array-like, Tensorfow variable, scalar or Initializer
        Initializer for initial hidden state (:math:`h_0`).
        Defaults to :class:`Constant(0) <neupy.init.Constant>`.

    {BaseRNNLayer.only_return_final}

    backwards : bool
        If ``True``, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`. Defaults to ``False``.

    unroll_scan : bool
        If ``True`` the recursion is unrolled instead of using scan.
        For some graphs this gives a significant speed up but it
        might also consume more memory. When ``unroll_scan=True``,
        backpropagation always includes the full sequence, so
        ``n_gradient_steps`` must be set to ``-1`` and the input
        sequence length must be known at compile time (i.e.,
        cannot be given as ``None``). Defaults to ``False``.

    {BaseLayer.Parameters}

    Notes
    -----
    Code was adapted from the
    `Lasagne <https://github.com/Lasagne/Lasagne>`_ library.

    Examples
    --------

    Sequence classification

    .. code-block:: python

        from neupy import layers, algorithms

        n_time_steps = 40
        n_categories = 20
        embedded_size = 10

        network = algorithms.RMSProp(
            [
                layers.Input(n_time_steps),
                layers.Embedding(n_categories, embedded_size),
                layers.GRU(20),
                layers.Sigmoid(1),
            ]
        )
    """
    input_weights = ParameterProperty(default=init.HeNormal())
    hidden_weights = ParameterProperty(default=init.HeNormal())
    biases = ParameterProperty(default=init.Constant(0))

    activation_functions = MultiCallableProperty(
        default=dict(
            resetgate=tf.nn.sigmoid,
            updategate=tf.nn.sigmoid,
            hidden_update=tf.tanh,
        )
    )

    learn_init = Property(default=False, expected_type=bool)
    hidden_init = ParameterProperty(default=init.Constant(0))

    backwards = Property(default=False, expected_type=bool)
    unroll_scan = Property(default=False, expected_type=bool)
    gradient_clipping = NumberProperty(default=0, minval=0)

    def initialize(self):
        super(GRU, self).initialize()
        n_inputs = np.prod(self.input_shape[1:])

        self.input_weights = self.add_parameter(
            value=self.input_weights,
            name='input_weights',
            shape=(n_inputs, 3 * self.size),
        )
        self.hidden_weights = self.add_parameter(
            value=self.hidden_weights,
            name='hidden_weights',
            shape=(self.size, 3 * self.size),
        )
        self.biases = self.add_parameter(
            value=self.biases, name='biases',
            shape=(3 * self.size,),
        )

        self.add_parameter(
            value=self.hidden_init,
            shape=(1, self.size),
            name="hidden_init",
            trainable=self.learn_init
        )

    def output(self, input_value):
        # Because scan iterates over the first dimension we
        # dimshuffle to (n_time_steps, n_batch, n_features)
        input_value = tf.transpose(input_value, [1, 0, 2])
        input_shape = tf.shape(input_value)
        n_batch = input_shape[1]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def one_gru_step(states, input_n):
            with tf.name_scope('gru-cell'):
                hid_previous, = states
                input_n = tf.matmul(input_n, self.input_weights) + self.biases

                # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1},
                # and W_{hc} h_{t - 1}
                hid_input = tf.matmul(hid_previous, self.hidden_weights)

                if self.gradient_clipping != 0:
                    input_n = clip_gradient(input_n, self.gradient_clipping)
                    hid_input = clip_gradient(
                        hid_input, self.gradient_clipping)

                hid_resetgate, hid_updategate, hid_hidden = tf.split(
                    hid_input, 3, axis=1)

                in_resetgate, in_updategate, in_hidden = tf.split(
                    input_n, 3, axis=1)

                # Reset and update gates
                resetgate = self.activation_functions.resetgate(
                    hid_resetgate + in_resetgate)

                updategate = self.activation_functions.updategate(
                    hid_updategate + in_updategate)

                # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
                hidden_update = in_hidden + resetgate * hid_hidden

                if self.gradient_clipping != 0:
                    hidden_update = clip_gradient(
                        hidden_update, self.gradient_clipping)

                hidden_update = self.activation_functions.hidden_update(
                    hidden_update)

                # Compute (1 - u_t)h_{t - 1} + u_t c_t
                return [
                    hid_previous - updategate * (hid_previous - hidden_update)
                ]

        hidden_init = tf.tile(self.hidden_init, (n_batch, 1))
        sequence = input_value

        if self.backwards:
            sequence = tf.reverse(sequence, axis=[0])

        if self.unroll_scan:
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=one_gru_step,
                sequence=sequence,
                outputs_info=[hidden_init]
            )
        else:
            hid_out, = tf.scan(
                fn=one_gru_step,
                elems=input_value,
                initializer=[hidden_init],
                name='gru-scan',
            )

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            return hid_out[-1]

        # if scan is backward reverse the output
        if self.backwards:
            hid_out = tf.reverse(hid_out, axis=[0])

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        hid_out = tf.transpose(hid_out, [1, 0, 2])
        return hid_out
