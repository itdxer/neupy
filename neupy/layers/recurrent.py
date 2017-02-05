import theano
import theano.tensor as T
import numpy as np

from neupy import init
from neupy.utils import AttributeKeyDict, as_tuple
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

    outputs = []
    prev_vals = outputs_info
    for i in counter:
        step_input = [s[i] for s in sequences] + prev_vals + non_sequences
        output = fn(*step_input)

        if not isinstance(output, list):
            output = [output]

        outputs.append(output)
        prev_vals = output

    # iterate over each scan output and convert it to
    # same format as scan:
    # [[output11, output12, ..., output1n],
    #  [output21, output22, ..., output2n],
    #  ...]
    output_scan = []
    for i in range(len(outputs[0])):
        l = map(lambda x: x[i], outputs)
        output_scan.append(T.stack(*l))

    return output_scan


class MultiParameterProperty(ParameterProperty):
    expected_type = as_tuple(init.Initializer, dict)

    def validate(self, value):
        super(MultiParameterProperty, self).validate(value)

        if isinstance(value, dict):
            for key in value:
                if key not in self.default:
                    valid_keys = ', '.join(self.default.keys())
                    raise ValueError("Parameter `{}` has invalid key: `{}`. "
                                     "Valid keys are: {}"
                                     "".format(self.name, key, valid_keys))

    def __set__(self, instance, value):
        self.validate(value)

        if isinstance(value, init.Initializer):
            # All keys will have the same initializer
            dict_value = dict.fromkeys(self.default.keys())

            for key in dict_value:
                dict_value[key] = value

            value = dict_value

        default_value = self.default.copy()
        default_value.update(value)
        value = default_value

        value = AttributeKeyDict(value)
        instance.__dict__[self.name] = value


class MultiCallableProperty(MultiParameterProperty):
    expected_type = as_tuple(dict)

    def validate(self, value):
        if not isinstance(value, self.expected_type):
            raise TypeError("Parameter `{}` should be a dictionary, "
                            "got `{!r}`".format(self.name, type(value)))

        for key, func in value.items():
            if not callable(func):
                raise ValueError("Values for the `{}` parameter should be "
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
        sequence is desired). In this case, Theano makes an
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

        if n_input_dims < 3:
            raise LayerConnectionError(
                "{} layer was expected input with at least three "
                "dimensions, got input with {} dimensions instead"
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

    weights : dict or Initializer
        Weight parameters for different gates.
        Defaults to :class:`XavierUniform() <neupy.init.XavierUniform>`.

        - In case if application requires the same initialization method
          for all weights, then it's possible to specify initialization
          method that would be automaticaly applied to all weight
          parameters in the LSTM layer.

          .. code-block:: python

              layers.LSTM(2, weights=init.Normal(0.1))

        - In case if application requires different initialization
          values for different weights then it's possible to specify
          an exact weight by name.

          .. code-block:: python

              dict(
                  weight_in_to_ingate=init.XavierUniform(),
                  weight_hid_to_ingate=init.XavierUniform(),
                  weight_cell_to_ingate=init.XavierUniform(),

                  weight_in_to_forgetgate=init.XavierUniform(),
                  weight_hid_to_forgetgate=init.XavierUniform(),
                  weight_cell_to_forgetgate=init.XavierUniform(),

                  weight_in_to_outgate=init.XavierUniform(),
                  weight_hid_to_outgate=init.XavierUniform(),
                  weight_cell_to_outgate=init.XavierUniform(),

                  weight_in_to_cell=init.XavierUniform(),
                  weight_hid_to_cell=init.XavierUniform(),
              )

          If application requires modification to only one (or multiple)
          parameter then it's better to specify the one that you need to
          modify and ignore other parameters

          .. code-block:: python

              dict(weight_in_to_ingate=init.Normal(0.1))

          Other parameters like ``weight_cell_to_outgate`` will be
          equal to their default values.

    biases : dict or Initializer
        Bias parameters for different gates.
        Defaults to :class:`Constant(0) <neupy.init.Constant>`.

        - In case if application requires the same initialization method
          for all biases, then it's possible to specify initialization
          method that would be automaticaly applied to all bias parameters
          in the LSTM layer.

          .. code-block:: python

              layers.LSTM(2, biases=init.Constant(1))

        - In case if application requires different initialization
          values for different weights then it's possible to specify
          an exact weight by name.

          .. code-block:: python

              dict(
                  bias_ingate=init.Constant(0),
                  bias_forgetgate=init.Constant(0),
                  bias_cell=init.Constant(0),
                  bias_outgate=init.Constant(0),
              )

          If application requires modification to only one (or multiple)
          parameter then it's better to specify the one that you need to
          modify and ignore other parameters

          .. code-block:: python

              dict(bias_ingate=init.Constant(1))

          Other parameters like ``bias_cell`` will be
          equal to their default values.

    activation_functions : dict, callable
        Activation functions for different gates. Defaults to:

        .. code-block:: python

            # import theano.tensor as T
            dict(
                ingate=T.nnet.sigmoid,
                forgetgate=T.nnet.sigmoid,
                outgate=T.nnet.sigmoid,
                cell=T.tanh,
            )

        If application requires modification to only one parameter
        then it's better to specify the one that you need to modify
        and ignore other parameters

        .. code-block:: python

            dict(ingate=T.tanh)

        Other parameters like ``forgetgate`` or ``outgate`` will be
        equal to their default values.

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
        from :math:`x_1` to :math:`x_n`. Defaults to ``False``

    {BaseRNNLayer.only_return_final}

    precompute_input : bool
        if ``True``, precompute ``input_to_hid`` before iterating
        through the sequence. This can result in a speed up at the
        expense of an increase in memory usage.
        Defaults to ``True``.

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

    gradient_clipping : flaot or int
        If nonzero, the gradient messages are clipped to the
        given value during the backward pass. Defaults to ``0``.

    n_gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If ``-1``, backpropagate through the entire sequence.
        Defaults to ``-1``.

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
    weights = MultiParameterProperty(
        default=dict(
            weight_in_to_ingate=init.XavierUniform(),
            weight_hid_to_ingate=init.XavierUniform(),
            weight_cell_to_ingate=init.XavierUniform(),

            weight_in_to_forgetgate=init.XavierUniform(),
            weight_hid_to_forgetgate=init.XavierUniform(),
            weight_cell_to_forgetgate=init.XavierUniform(),

            weight_in_to_outgate=init.XavierUniform(),
            weight_hid_to_outgate=init.XavierUniform(),
            weight_cell_to_outgate=init.XavierUniform(),

            weight_in_to_cell=init.XavierUniform(),
            weight_hid_to_cell=init.XavierUniform(),
        ))
    biases = MultiParameterProperty(
        default=dict(
            bias_ingate=init.Constant(0),
            bias_forgetgate=init.Constant(0),
            bias_cell=init.Constant(0),
            bias_outgate=init.Constant(0),
        ))
    activation_functions = MultiCallableProperty(
        default=dict(
            ingate=T.nnet.sigmoid,
            forgetgate=T.nnet.sigmoid,
            outgate=T.nnet.sigmoid,
            cell=T.tanh,
        ))

    learn_init = Property(default=False, expected_type=bool)
    cell_init = ParameterProperty(default=init.Constant(0))
    hid_init = ParameterProperty(default=init.Constant(0))

    unroll_scan = Property(default=False, expected_type=bool)
    backwards = Property(default=False, expected_type=bool)
    precompute_input = Property(default=True, expected_type=bool)
    peepholes = Property(default=False, expected_type=bool)

    n_gradient_steps = IntProperty(default=-1)
    gradient_clipping = NumberProperty(default=0, minval=0)

    def initialize(self):
        super(LSTM, self).initialize()

        n_inputs = np.prod(self.input_shape[1:])
        weights = self.weights
        biases = self.biases

        # Input gate parameters
        self.weight_in_to_ingate = self.add_parameter(
            value=weights.weight_in_to_ingate,
            name='weight_in_to_ingate',
            shape=(n_inputs, self.size))
        self.weight_hid_to_ingate = self.add_parameter(
            value=weights.weight_hid_to_ingate,
            name='weight_hid_to_ingate',
            shape=(self.size, self.size))
        self.bias_ingate = self.add_parameter(
            value=biases.bias_ingate, name='bias_ingate',
            shape=(self.size,))

        # Forget gate parameters
        self.weight_in_to_forgetgate = self.add_parameter(
            value=weights.weight_in_to_forgetgate,
            name='weight_in_to_forgetgate',
            shape=(n_inputs, self.size))
        self.weight_hid_to_forgetgate = self.add_parameter(
            value=weights.weight_hid_to_forgetgate,
            name='weight_hid_to_forgetgate',
            shape=(self.size, self.size))
        self.bias_forgetgate = self.add_parameter(
            value=biases.bias_forgetgate, name='bias_forgetgate',
            shape=(self.size,))

        # Cell parameters
        self.weight_in_to_cell = self.add_parameter(
            value=weights.weight_in_to_cell,
            name='weight_in_to_cell',
            shape=(n_inputs, self.size))
        self.weight_hid_to_cell = self.add_parameter(
            value=weights.weight_hid_to_cell,
            name='weight_hid_to_cell',
            shape=(self.size, self.size))
        self.bias_cell = self.add_parameter(
            value=biases.bias_cell, name='bias_cell',
            shape=(self.size,))

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.weight_cell_to_ingate = self.add_parameter(
                value=weights.weight_cell_to_ingate,
                name='weight_cell_to_ingate',
                shape=(self.size,))
            self.weight_cell_to_forgetgate = self.add_parameter(
                value=weights.weight_cell_to_forgetgate,
                name='weight_cell_to_forgetgate',
                shape=(self.size,))
            self.weight_cell_to_outgate = self.add_parameter(
                value=weights.weight_cell_to_outgate,
                name='weight_cell_to_outgate',
                shape=(self.size,))

        # Output gate parameters
        self.weight_in_to_outgate = self.add_parameter(
            value=weights.weight_in_to_outgate,
            name='weight_in_to_outgate',
            shape=(n_inputs, self.size))
        self.weight_hid_to_outgate = self.add_parameter(
            value=weights.weight_hid_to_outgate,
            name='weight_hid_to_outgate',
            shape=(self.size, self.size))
        self.bias_outgate = self.add_parameter(
            value=biases.bias_outgate, name='bias_outgate',
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
            # weight_in_stacked is (n_features, 4 * num_units).
            # Input: (n_time_steps, n_batch, 4 * num_units).
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
            ingate = self.activation_functions.ingate(ingate)
            forgetgate = self.activation_functions.forgetgate(forgetgate)
            cell_input = self.activation_functions.cell(cell_input)

            # Compute new cell value
            cell = forgetgate * cell_previous + ingate * cell_input

            if self.peepholes:
                outgate += cell * self.weight_cell_to_outgate

            outgate = self.activation_functions.outgate(outgate)

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


class GRU(BaseRNNLayer):
    """
    Gated Recurrent Unit (GRU) Layer.

    Parameters
    ----------
    {BaseRNNLayer.size}

    weights : dict or Initializer
        Weight parameters for different gates.
        Defaults to :class:`XavierUniform() <neupy.init.XavierUniform>`.

        - In case if application requires the same initialization method
          for all weights, then it's possible to specify initialization
          method that would be automaticaly applied to all weight
          parameters in the GRU layer.

          .. code-block:: python

              layers.GRU(2, weights=init.Normal(0.1))

        - In case if application requires different initialization
          values for different weights then it's possible to specify
          an exact weight by name.

          .. code-block:: python

              dict(
                  weight_in_to_updategate=init.XavierUniform(),
                  weight_hid_to_updategate=init.XavierUniform(),

                  weight_in_to_resetgate=init.XavierUniform(),
                  weight_hid_to_resetgate=init.XavierUniform(),

                  weight_in_to_hidden_update=init.XavierUniform(),
                  weight_hid_to_hidden_update=init.XavierUniform(),
              )

          If application requires modification to only one (or multiple)
          parameter then it's better to specify the one that you need to
          modify and ignore other parameters

          .. code-block:: python

              dict(weight_in_to_updategate=init.Normal(0.1))

          Other parameters like ``weight_in_to_resetgate`` will be
          equal to their default values.

    biases : dict or Initializer
        Bias parameters for different gates.
        Defaults to :class:`Constant(0) <neupy.init.Constant>`.

        - In case if application requires the same initialization method
          for all biases, then it's possible to specify initialization
          method that would be automaticaly applied to all bias parameters
          in the GRU layer.

          .. code-block:: python

              layers.GRU(2, biases=init.Constant(1))

        - In case if application requires different initialization
          values for different weights then it's possible to specify
          an exact weight by name.

          .. code-block:: python

              dict(
                  bias_updategate=init.Constant(0),
                  bias_resetgate=init.Constant(0),
                  bias_hidden_update=init.Constant(0),
              )

          If application requires modification to only one (or multiple)
          parameter then it's better to specify the one that you need to
          modify and ignore other parameters

          .. code-block:: python

              dict(bias_resetgate=init.Constant(1))

          Other parameters like ``bias_updategate`` will be
          equal to their default values.

    activation_functions : dict, callable
        Activation functions for different gates. Defaults to:

        .. code-block:: python

            # import theano.tensor as T
            dict(
                resetgate=T.nnet.sigmoid,
                updategate=T.nnet.sigmoid,
                hidden_update=T.tanh,
            )

        If application requires modification to only one parameter
        then it's better to specify the one that you need to modify
        and ignore other parameters

        .. code-block:: python

            dict(resetgate=T.tanh)

        Other parameters like ``updategate`` or ``hidden_update``
        will be equal to their default values.

    learn_init : bool
        If ``True``, make ``hid_init`` trainable variable.
        Defaults to ``False``.

    hid_init : array-like, Theano variable, scalar or Initializer
        Initializer for initial hidden state (:math:`h_0`).
        Defaults to :class:`Constant(0) <neupy.init.Constant>`.

    {BaseRNNLayer.only_return_final}

    backwards : bool
        If ``True``, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`. Defaults to ``False``.

    precompute_input : bool
        if ``True``, precompute ``input_to_hid`` before iterating
        through the sequence. This can result in a speed up at the
        expense of an increase in memory usage.
        Defaults to ``True``.

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
    weights = MultiParameterProperty(
        default=dict(
            weight_in_to_updategate=init.XavierUniform(),
            weight_hid_to_updategate=init.XavierUniform(),

            weight_in_to_resetgate=init.XavierUniform(),
            weight_hid_to_resetgate=init.XavierUniform(),

            weight_in_to_hidden_update=init.XavierUniform(),
            weight_hid_to_hidden_update=init.XavierUniform(),
        ))
    biases = MultiParameterProperty(
        default=dict(
            bias_updategate=init.Constant(0),
            bias_resetgate=init.Constant(0),
            bias_hidden_update=init.Constant(0),
        ))
    activation_functions = MultiCallableProperty(
        default=dict(
            resetgate=T.nnet.sigmoid,
            updategate=T.nnet.sigmoid,
            hidden_update=T.tanh,
        ))

    learn_init = Property(default=False, expected_type=bool)
    hid_init = ParameterProperty(default=init.Constant(0))

    backwards = Property(default=False, expected_type=bool)
    unroll_scan = Property(default=False, expected_type=bool)
    precompute_input = Property(default=True, expected_type=bool)

    n_gradient_steps = IntProperty(default=-1)
    gradient_clipping = NumberProperty(default=0, minval=0)

    def initialize(self):
        super(GRU, self).initialize()

        n_inputs = np.prod(self.input_shape[1:])
        weights = self.weights
        biases = self.biases

        # Update gate parameters
        self.weight_in_to_updategate = self.add_parameter(
            value=weights.weight_in_to_updategate,
            name='weight_in_to_updategate',
            shape=(n_inputs, self.size))
        self.weight_hid_to_updategate = self.add_parameter(
            value=weights.weight_hid_to_updategate,
            name='weight_hid_to_updategate',
            shape=(self.size, self.size))
        self.bias_updategate = self.add_parameter(
            value=biases.bias_updategate, name='bias_updategate',
            shape=(self.size,))

        # Reset gate parameters
        self.weight_in_to_resetgate = self.add_parameter(
            value=weights.weight_in_to_resetgate,
            name='weight_in_to_resetgate',
            shape=(n_inputs, self.size))
        self.weight_hid_to_resetgate = self.add_parameter(
            value=weights.weight_hid_to_resetgate,
            name='weight_hid_to_resetgate',
            shape=(self.size, self.size))
        self.bias_resetgate = self.add_parameter(
            value=biases.bias_resetgate, name='bias_forgetgate',
            shape=(self.size,))

        # Hidden update gate parameters
        self.weight_in_to_hidden_update = self.add_parameter(
            value=weights.weight_in_to_hidden_update,
            name='weight_in_to_hidden_update',
            shape=(n_inputs, self.size))
        self.weight_hid_to_hidden_update = self.add_parameter(
            value=weights.weight_hid_to_hidden_update,
            name='weight_hid_to_hidden_update',
            shape=(self.size, self.size))
        self.bias_hidden_update = self.add_parameter(
            value=biases.bias_hidden_update, name='bias_hidden_update',
            shape=(self.size,))

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

        # Stack input weight matrices into a (num_inputs, 3 * num_units)
        # matrix, which speeds up computation
        weight_in_stacked = T.concatenate([
            self.weight_in_to_updategate,
            self.weight_in_to_resetgate,
            self.weight_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        weight_hid_stacked = T.concatenate([
            self.weight_hid_to_updategate,
            self.weight_hid_to_resetgate,
            self.weight_hid_to_hidden_update], axis=1)

        # Stack biases into a (3 * num_units) vector
        bias_stacked = T.concatenate([
            self.bias_updategate,
            self.bias_resetgate,
            self.bias_hidden_update], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # weight_in_stacked is (n_features, 3 * num_units).
            # Input: (n_time_steps, n_batch, 3 * num_units).
            input_value = T.dot(input_value, weight_in_stacked) + bias_stacked

        # When theano.scan calls step, input_n will be
        # (n_batch, 3 * num_units). We define a slicing function
        # that extract the input to each GRU gate
        def slice_w(x, n):
            s = x[:, n * self.size:(n + 1) * self.size]
            if self.size == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def one_gru_step(input_n, hid_previous, *args):
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1},
            # and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, weight_hid_stacked)

            if self.gradient_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n,
                    -self.gradient_clipping,
                    self.gradient_clipping)

                hid_input = theano.gradient.grad_clip(
                    hid_input,
                    -self.gradient_clipping,
                    self.gradient_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u,
                # and W_{xc}x_t + b_c
                input_n = T.dot(input_n, weight_in_stacked) + bias_stacked

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            resetgate = self.activation_functions.resetgate(resetgate)

            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            updategate = self.activation_functions.updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate * hidden_update_hid

            if self.gradient_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update,
                    -self.gradient_clipping,
                    self.gradient_clipping)

            hidden_update = self.activation_functions.hidden_update(
                hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate) * hid_previous + updategate * hidden_update
            return hid

        hid_init = T.dot(T.ones((n_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_sequences = [weight_hid_stacked]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_sequences += [weight_in_stacked, bias_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            n_time_steps = self.input_shape[0]

            # Explicitly unroll the recurrence instead of using scan
            hid_out, = unroll_scan(
                fn=one_gru_step,
                sequences=[input_value],
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_sequences,
                n_steps=n_time_steps)

        else:
            # Scan op iterates over first dimension of input and
            # repeatedly applies the step function
            hid_out, _ = theano.scan(
                fn=one_gru_step,
                sequences=[input_value],
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_sequences,
                truncate_gradient=self.n_gradient_steps,
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
