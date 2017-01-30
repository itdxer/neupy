import theano
import theano.tensor as T
import numpy as np

from neupy import init
from neupy.utils import as_tuple
from neupy.exceptions import LayerConnectionError
from neupy.core.properties import (IntProperty, Property, NumberProperty,
                                   ParameterProperty)
from .base import BaseLayer


__all__ = ('LSTM', 'GRU')


def unroll_scan(fn, sequences, outputs_info, non_sequences, n_steps,
                go_backwards=False):
    """
    Helper function to unroll for loops. Can be used to unroll theano.scan.
    The parameter names are identical to theano.scan, please refer to here
    for more information.
    Note that this function does not support the truncate_gradient
    setting from theano.scan.

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

    non_sequences: list of TensorVariables
        List of theano.shared variables that are used in the step function.

    n_steps: int
        Number of steps to unroll.

    go_backwards: bool
        If ``True`` the recursion starts at sequences[-1] and
        iterates backwards.

    Returns
    -------
    List of TensorVariables. Each element in the list gives
    the recurrent values at each time step.
    """
    # When backwards reverse the recursion direction
    counter = range(n_steps)
    if go_backwards:
        counter = counter[::-1]

    output = []
    prev_vals = outputs_info
    for i in counter:
        step_input = [s[i] for s in sequences] + prev_vals + non_sequences
        prev_vals = out_ = fn(*step_input)
        output.append(out_)

    # iterate over each scan output and convert it to same format as scan:
    # [[output11, output12,...output1n],
    #  [output21, output22,...output2n],
    #  ...]
    output_scan = []
    for i in range(len(output[0])):
        l = map(lambda x: x[i], output)
        output_scan.append(T.stack(*l))

    return output_scan


class LSTM(BaseLayer):
    """
    Long Short Term Memory (LSTM) Layer.

    Parameters
    ----------
    size : int
        Number of hidden units in the network.

    learn_init : bool
        If ``True``, make ``cell_init`` and ``hid_init`` trainable
        variables. Defaults to ``False``.

    cell_init : array-like, Theano variable, scalar or Initializer
        Initializer for initial cell state (:math:`c_0`).
        Defaults to :class:`Constant(0) <neupy.init.Constant>`.

    hid_init : array-like, Theano variable, scalar or Initializer
        Initializer for initial hidden state (:math:`h_0`).
        Defaults to :class:`Constant(0) <neupy.init.Constant>`.

    backwards : bool
        If ``True``, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.

    only_return_final : bool
        If ``True``, only return the final sequential output
        (e.g. for tasks where a single target value for the entire
        sequence is desired). In this case, Theano makes an
        optimization which saves memory. Defaults to ``True``.

    precompute_input : bool
        Defaults to ``True``.

    peepholes : bool
        Defaults to ``False``.

    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When ``unroll_scan=True``, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).

    grad_clipping : flaot or int
        If nonzero, the gradient messages are clipped to the
        given value during the backward pass. Defaults to ``0``.

    n_gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If ``-1``, backpropagate through the entire sequence.
        Defaults to ``-1``.

    Notes
    -----
    Code was adapted from the
    `Lasagne <https://github.com/Lasagne/Lasagne>`_ library.
    """
    size = IntProperty(minval=1)

    ingate = Property()
    forgetgate = Property()
    outgate = Property()
    cell = Property()

    learn_init = Property(default=False, expected_type=bool)
    cell_init = ParameterProperty(default=init.Constant(0))
    hid_init = ParameterProperty(default=init.Constant(0))

    only_return_final = Property(default=True, expected_type=bool)
    unroll_scan = Property(default=False, expected_type=bool)
    backwards = Property(default=False, expected_type=bool)
    precompute_input = Property(default=True, expected_type=bool)
    peepholes = Property(default=False, expected_type=bool)

    n_gradient_steps = IntProperty(default=-1)
    gradient_clipping = NumberProperty(default=0, minval=0)

    def __init__(self, size, **kwargs):
        super(LSTM, self).__init__(size=size, **kwargs)

    def validate(self, input_shape):
        n_input_dims = len(input_shape) + 1  # +1 for batch dimension
        if n_input_dims < 3:
            raise LayerConnectionError(
                "LSTM was expected input with at least three dimensions, "
                "got input with {} dimensions instead".format(n_input_dims))

    @property
    def output_shape(self):
        if self.only_return_final:
            return as_tuple(self.size)

        n_time_steps = self.input_shape[0]
        return as_tuple(n_time_steps, self.size)

    def initialize(self):
        super(LSTM, self).initialize()

        n_inputs = np.prod(self.input_shape[1:])

        # Input gate parameters
        self.weight_in_to_ingate = self.add_parameter(
            value=init.HeUniform(), name='weight_in_to_ingate',
            shape=(n_inputs, self.size))
        self.weight_hid_to_ingate = self.add_parameter(
            value=init.HeUniform(), name='weight_hid_to_ingate',
            shape=(self.size, self.size))
        self.bias_ingate = self.add_parameter(
            value=init.Constant(0), name='bias_ingate',
            shape=(self.size,))

        # Forget gate parameters
        self.weight_in_to_forgetgate = self.add_parameter(
            value=init.HeUniform(), name='weight_in_to_forgetgate',
            shape=(n_inputs, self.size))
        self.weight_hid_to_forgetgate = self.add_parameter(
            value=init.HeUniform(), name='weight_hid_to_forgetgate',
            shape=(self.size, self.size))
        self.bias_forgetgate = self.add_parameter(
            value=init.Constant(0), name='bias_forgetgate',
            shape=(self.size,))

        # Cell parameters
        self.weight_in_to_cell = self.add_parameter(
            value=init.HeUniform(), name='weight_in_to_cell',
            shape=(n_inputs, self.size))
        self.weight_hid_to_cell = self.add_parameter(
            value=init.HeUniform(), name='weight_hid_to_cell',
            shape=(self.size, self.size))
        self.bias_cell = self.add_parameter(
            value=init.Constant(0), name='bias_cell',
            shape=(self.size,))

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.weight_cell_to_ingate = self.add_parameter(
                value=init.HeUniform(), name='weight_cell_to_ingate',
                shape=(self.size,))
            self.weight_cell_to_forgetgate = self.add_parameter(
                value=init.HeUniform(), name='weight_cell_to_forgetgate',
                shape=(self.size,))
            self.weight_cell_to_outgate = self.add_parameter(
                value=init.HeUniform(), name='weight_cell_to_outgate',
                shape=(self.size,))

        # Output gate parameters
        self.weight_in_to_outgate = self.add_parameter(
            value=init.HeUniform(), name='weight_in_to_outgate',
            shape=(n_inputs, self.size))
        self.weight_hid_to_outgate = self.add_parameter(
            value=init.HeUniform(), name='weight_hid_to_outgate',
            shape=(self.size, self.size))
        self.bias_outgate = self.add_parameter(
            value=init.Constant(0), name='bias_outgate',
            shape=(self.size,))

        # Initialization parameters
        self.add_parameter(value=self.cell_init, shape=(1, self.size),
                           name="cell_init", trainable=self.learn_init)
        self.add_parameter(value=self.hid_init, shape=(1, self.size),
                           name="hid_init", trainable=self.learn_init)

    def output(self, input_value):
        # Treat all dimensions after the second as flattened
        # feature dimensions
        if input_value.ndim > 3:
            input_value = T.flatten(input_value, 3)

        # Because scan iterates over the first dimension we
        # dimshuffle to (n_time_steps, n_batch, n_features)
        input_value = input_value.dimshuffle(1, 0, 2)
        seq_len, n_batch, _ = input_value.shape

        # Stack input weight matrices into a (num_inputs, 4 * num_units)
        # matrix, which speeds up computation
        weight_in_stacked = T.concatenate([
            self.weight_in_to_ingate,
            self.weight_in_to_forgetgate,
            self.weight_in_to_cell,
            self.weight_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        weight_hid_stacked = T.concatenate([
            self.weight_hid_to_ingate,
            self.weight_hid_to_forgetgate,
            self.weight_hid_to_cell,
            self.weight_hid_to_outgate], axis=1)

        # Stack biases into a (4 * num_units) vector
        bias_stacked = T.concatenate([
            self.bias_ingate,
            self.bias_forgetgate,
            self.bias_cell,
            self.bias_outgate], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # weight_in_stacked is (n_features, 4 * num_units). input is then
            # (n_time_steps, n_batch, 4 * num_units).
            input_value = T.dot(input_value, weight_in_stacked) + bias_stacked

        # When theano.scan calls step, input_n will be
        # (n_batch, 4 * num_units). We define a slicing function
        # that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n * self.size:(n + 1) * self.size]

        def one_lstm_step(input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, weight_in_stacked) + bias_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, weight_hid_stacked)

            # Clip gradients
            if self.gradient_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.gradient_clipping, self.gradient_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous * self.weight_cell_to_ingate
                forgetgate += cell_previous * self.weight_cell_to_forgetgate

            # Apply nonlinearities
            ingate = T.nnet.sigmoid(ingate)
            forgetgate = T.nnet.sigmoid(forgetgate)
            cell_input = T.tanh(cell_input)

            # Compute new cell value
            cell = forgetgate * cell_previous + ingate * cell_input

            if self.peepholes:
                outgate += cell * self.weight_cell_to_outgate
            outgate = T.nnet.sigmoid(outgate)

            # Compute new hidden unit activation
            hid = outgate * T.tanh(cell)
            return [cell, hid]

        ones = T.ones((n_batch, 1))
        cell_init = T.dot(ones, self.cell_init)
        hid_init = T.dot(ones, self.hid_init)

        non_sequences = [weight_hid_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_sequences += [weight_in_stacked, bias_stacked]

        # The "peephole" weight matrices are only used
        # when self.peepholes=True
        if self.peepholes:
            non_sequences += [self.weight_cell_to_ingate,
                              self.weight_cell_to_forgetgate,
                              self.weight_cell_to_outgate]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            n_time_steps = self.input_shape[0]

            # Explicitly unroll the recurrence instead of using scan
            _, hid_out = unroll_scan(
                fn=one_lstm_step,
                sequences=[input_value],
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_sequences,
                n_steps=n_time_steps)

        else:
            (_, hid_out), _ = theano.scan(
                fn=one_lstm_step,
                sequences=input_value,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.n_gradient_steps,
                non_sequences=non_sequences,
                strict=True)

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            return hid_out[-1]

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        hid_out = hid_out.dimshuffle(1, 0, 2)

        # if scan is backward reverse the output
        if self.backwards:
            hid_out = hid_out[:, ::-1]

        return hid_out


class GRU(BaseLayer):
    """
    Gated Recurrent Unit (GRU) Layer.
    """
