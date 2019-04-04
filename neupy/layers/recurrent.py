import types

import tensorflow as tf

from neupy import init
from neupy.utils import tensorflow_session
from neupy.exceptions import LayerConnectionError
from neupy.core.properties import (
    IntProperty, Property,
    NumberProperty, ParameterProperty,
)
from .base import BaseLayer


__all__ = ('LSTM', 'GRU')


def clip_gradient(value, clip_value):
    if not hasattr(clip_gradient, 'added_gradients'):
        clip_gradient.added_gradients = set()

    session = tensorflow_session()
    graph = session.graph
    operation_name = "ClipGradient-" + str(clip_value)

    if operation_name not in clip_gradient.added_gradients:
        # Make sure that we won't create the same operation twice.
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


class BaseRNNLayer(BaseLayer):
    """
    Base class for the recurrent layers

    Parameters
    ----------
    n_units : int
        Number of hidden units in the layer.

    only_return_final : bool
        If ``True``, only return the final sequential output
        (e.g. for tasks where a single target value for the entire
        sequence is desired). In this case, Tensorfow makes an
        optimization which saves memory. Defaults to ``True``.

    {BaseLayer.name}
    """
    n_units = IntProperty(minval=1)
    only_return_final = Property(expected_type=bool)

    def __init__(self, n_units, only_return_final=True, name=None):
        super(BaseRNNLayer, self).__init__(name=name)
        self.only_return_final = only_return_final
        self.n_units = n_units

    def fail_if_shape_invalid(self, input_shape):
        if input_shape and input_shape.ndims != 3:
            clsname = self.__class__.__name__
            raise LayerConnectionError(
                "{} layer was expected input with three dimensions, "
                "but got input with {} dimensions instead. Layer: {}"
                "".format(clsname, input_shape.ndims, self))

    def get_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        n_samples = input_shape[0]

        self.fail_if_shape_invalid(input_shape)

        if self.only_return_final:
            return tf.TensorShape((n_samples, self.n_units))

        n_time_steps = input_shape[1]
        return tf.TensorShape((n_samples, n_time_steps, self.n_units))


class LSTM(BaseRNNLayer):
    """
    Long Short Term Memory (LSTM) Layer.

    Parameters
    ----------
    {BaseRNNLayer.n_units}

    {BaseRNNLayer.only_return_final}

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

    ingate : function
        Activation function for the input gate.
        Defaults to ``tf.nn.sigmoid``.

    forgetgate : function
        Activation function for the forget gate.
        Defaults to ``tf.nn.sigmoid``.

    outgate : function
        Activation function for the output gate.
        Defaults to ``tf.nn.sigmoid``.

    cell : function
        Activation function for the cell.
        Defaults to ``tf.tanh``.

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

    {BaseLayer.name}

    Notes
    -----
    Code was adapted from the
    `Lasagne <https://github.com/Lasagne/Lasagne>`_ library.

    Examples
    --------
    Sequence classification

    >>> from neupy.layers import *
    >>>
    >>> n_time_steps = 40
    >>> n_categories = 20
    >>> embedded_size = 10
    >>>
    >>> network = join(
    ...     Input(n_time_steps),
    ...     Embedding(n_categories, embedded_size),
    ...     LSTM(20),
    ...     Sigmoid(1),
    ... )
    >>> network
    (?, 40) -> [... 4 layers ...] -> (?, 1)
    """
    input_weights = ParameterProperty()
    hidden_weights = ParameterProperty()
    cell_weights = ParameterProperty()
    biases = ParameterProperty()

    ingate = Property(expected_type=types.FunctionType)
    forgetgate = Property(expected_type=types.FunctionType)
    outgate = Property(expected_type=types.FunctionType)
    cell = Property(expected_type=types.FunctionType)

    learn_init = Property(expected_type=bool)
    cell_init = ParameterProperty()
    hidden_init = ParameterProperty()

    unroll_scan = Property(expected_type=bool)
    backwards = Property(expected_type=bool)
    peepholes = Property(expected_type=bool)
    gradient_clipping = NumberProperty(minval=0)

    def __init__(self, n_units, only_return_final=True,
                 # Trainable parameters
                 input_weights=init.HeNormal(),
                 hidden_weights=init.HeNormal(),
                 cell_weights=init.HeNormal(), biases=0,
                 # Activation functions
                 ingate=tf.nn.sigmoid, forgetgate=tf.nn.sigmoid,
                 outgate=tf.nn.sigmoid, cell=tf.tanh,
                 # Cell states
                 cell_init=0, hidden_init=0, learn_init=False,
                 # Misc
                 unroll_scan=False, backwards=False, peepholes=False,
                 gradient_clipping=0, name=None):

        super(LSTM, self).__init__(
            n_units=n_units,
            only_return_final=only_return_final,
            name=name,
        )

        self.input_weights = input_weights
        self.hidden_weights = hidden_weights
        self.cell_weights = cell_weights
        self.biases = biases

        self.ingate = ingate
        self.forgetgate = forgetgate
        self.outgate = outgate
        self.cell = cell

        self.learn_init = learn_init
        self.cell_init = cell_init
        self.hidden_init = hidden_init

        self.unroll_scan = unroll_scan
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_clipping = gradient_clipping

    def create_variables(self, input_shape):
        self.input_shape = input_shape
        self.input_weights = self.variable(
            value=self.input_weights,
            name='input_weights',
            shape=(input_shape[-1], 4 * self.n_units),
        )
        self.hidden_weights = self.variable(
            value=self.hidden_weights, name='hidden_weights',
            shape=(self.n_units, 4 * self.n_units),
        )
        self.biases = self.variable(
            value=self.biases, name='biases',
            shape=(4 * self.n_units,),
        )
        self.cell_init = self.variable(
            value=self.cell_init,
            shape=(1, self.n_units),
            name="cell_init",
            trainable=self.learn_init,
        )
        self.hidden_init = self.variable(
            value=self.hidden_init,
            shape=(1, self.n_units),
            name="hidden_init",
            trainable=self.learn_init,
        )

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.weight_cell_to_ingate = self.variable(
                value=self.cell_weights,
                name='weight_cell_to_ingate',
                shape=(self.n_units,))
            self.weight_cell_to_forgetgate = self.variable(
                value=self.cell_weights,
                name='weight_cell_to_forgetgate',
                shape=(self.n_units,))
            self.weight_cell_to_outgate = self.variable(
                value=self.cell_weights,
                name='weight_cell_to_outgate',
                shape=(self.n_units,))

    def output(self, input, **kwargs):
        # Because scan iterates over the first dimension we
        # dimshuffle to (n_time_steps, n_samples, n_features)
        input = tf.transpose(input, [1, 0, 2])

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
                ingate = self.ingate(ingate)
                forgetgate = self.forgetgate(forgetgate)
                cell_input = self.cell(cell_input)

                # Compute new cell value
                cell = forgetgate * cell_previous + ingate * cell_input

                if self.peepholes:
                    outgate += cell * self.weight_cell_to_outgate

                outgate = self.outgate(outgate)

                # Compute new hidden unit activation
                hid = outgate * tf.tanh(cell)
                return [cell, hid]

        input_shape = tf.shape(input)
        n_samples = input_shape[1]  # batch dim has been moved
        cell_init = tf.tile(self.cell_init, (n_samples, 1))
        hidden_init = tf.tile(self.hidden_init, (n_samples, 1))
        sequence = input

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
                elems=sequence,
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

        # dimshuffle back to (n_samples, n_time_steps, n_features))
        hid_out = tf.transpose(hid_out, [1, 0, 2])

        return hid_out


class GRU(BaseRNNLayer):
    """
    Gated Recurrent Unit (GRU) Layer.

    Parameters
    ----------
    {BaseRNNLayer.n_units}

    {BaseRNNLayer.only_return_final}

    input_weights : Initializer, ndarray
        Weight parameters for input connection.
        Defaults to :class:`HeNormal() <neupy.init.HeNormal>`.

    hidden_weights : Initializer, ndarray
        Weight parameters for hidden connection.
        Defaults to :class:`HeNormal() <neupy.init.HeNormal>`.

    biases : Initializer, ndarray
        Bias parameters for all gates.
        Defaults to :class:`Constant(0) <neupy.init.Constant>`.

    resetgate : function
        Activation function for the reset gate.
        Defaults to ``tf.nn.sigmoid``.

    updategate : function
        Activation function for the update gate.
        Defaults to ``tf.nn.sigmoid``.

    hidden_update : function
        Activation function for the hidden state update.
        Defaults to ``tf.tanh``.

    learn_init : bool
        If ``True``, make ``hidden_init`` trainable variable.
        Defaults to ``False``.

    hidden_init : array-like, Tensorfow variable, scalar or Initializer
        Initializer for initial hidden state (:math:`h_0`).
        Defaults to :class:`Constant(0) <neupy.init.Constant>`.

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

    {BaseLayer.name}

    Notes
    -----
    Code was adapted from the
    `Lasagne <https://github.com/Lasagne/Lasagne>`_ library.

    Examples
    --------
    Sequence classification

    >>> from neupy.layers import *
    >>>
    >>> n_time_steps = 40
    >>> n_categories = 20
    >>> embedded_size = 10
    >>>
    >>> network = join(
    ...     Input(n_time_steps),
    ...     Embedding(n_categories, embedded_size),
    ...     GRU(20),
    ...     Sigmoid(1),
    ... )
    >>> network
    (?, 40) -> [... 4 layers ...] -> (?, 1)
    """
    input_weights = ParameterProperty()
    hidden_weights = ParameterProperty()
    biases = ParameterProperty()

    resetgate = Property(expected_type=types.FunctionType)
    updategate = Property(expected_type=types.FunctionType)
    hidden_update = Property(expected_type=types.FunctionType)

    hidden_init = ParameterProperty()
    learn_init = Property(default=False, expected_type=bool)

    backwards = Property(expected_type=bool)
    unroll_scan = Property(expected_type=bool)
    gradient_clipping = NumberProperty(default=0, minval=0)

    def __init__(self, n_units, only_return_final=True,
                 # Trainable parameters
                 input_weights=init.HeNormal(),
                 hidden_weights=init.HeNormal(),
                 biases=0,
                 # Activation functions
                 resetgate=tf.nn.sigmoid,
                 updategate=tf.nn.sigmoid,
                 hidden_update=tf.tanh,
                 # Cell states
                 hidden_init=0, learn_init=False,
                 # Misc
                 unroll_scan=False, backwards=False,
                 gradient_clipping=0, name=None):

        super(GRU, self).__init__(
            n_units=n_units,
            only_return_final=only_return_final,
            name=name,
        )

        self.input_weights = input_weights
        self.hidden_weights = hidden_weights
        self.biases = biases

        self.resetgate = resetgate
        self.updategate = updategate
        self.hidden_update = hidden_update

        self.hidden_init = hidden_init
        self.learn_init = learn_init

        self.unroll_scan = unroll_scan
        self.backwards = backwards
        self.gradient_clipping = gradient_clipping

    def create_variables(self, input_shape):
        self.input_weights = self.variable(
            value=self.input_weights,
            name='input_weights',
            shape=(input_shape[-1], 3 * self.n_units),
        )
        self.hidden_weights = self.variable(
            value=self.hidden_weights,
            name='hidden_weights',
            shape=(self.n_units, 3 * self.n_units),
        )
        self.biases = self.variable(
            value=self.biases, name='biases',
            shape=(3 * self.n_units,),
        )
        self.hidden_init = self.variable(
            value=self.hidden_init,
            shape=(1, self.n_units),
            name="hidden_init",
            trainable=self.learn_init
        )

    def output(self, input, **kwargs):
        # Because scan iterates over the first dimension we
        # dimshuffle to (n_time_steps, n_samples, n_features)
        input = tf.transpose(input, [1, 0, 2])

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
                resetgate = self.resetgate(hid_resetgate + in_resetgate)
                updategate = self.updategate(hid_updategate + in_updategate)

                # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
                hidden_update = in_hidden + resetgate * hid_hidden

                if self.gradient_clipping != 0:
                    hidden_update = clip_gradient(
                        hidden_update, self.gradient_clipping)

                hidden_update = self.hidden_update(hidden_update)

                # Compute (1 - u_t)h_{t - 1} + u_t c_t
                return [
                    hid_previous - updategate * (hid_previous - hidden_update)
                ]

        input_shape = tf.shape(input)
        n_samples = input_shape[1]  # batch dim has been moved
        hidden_init = tf.tile(self.hidden_init, (n_samples, 1))
        sequence = input

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
                elems=sequence,
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

        # dimshuffle back to (n_samples, n_time_steps, n_features))
        hid_out = tf.transpose(hid_out, [1, 0, 2])
        return hid_out
